// Copyright © 2019-2025
// SPDX-License-Identifier: Apache-2.0

#include "opae_simx.h"
#include <VX_config.h>
#include <vortex_afu.h>
#include <arch.h>
#include <common.h>
#include <dram_sim.h>
#include <mem.h>
#include <mem_alloc.h>
#include <processor.h>
#include <util.h>
#include <cstring>

#include <future>
#include <mutex>
#include <unordered_map>


#define CACHE_BLOCK_SIZE 64
#define RAM_PAGE_SIZE 4096

#define PIN_BASE_ADDR 0x10000000ull
#define PINNED_MEM_SIZE 0x00ffffffull // 16 MiB of host-visible pinned RAM

#define MMIO_CMD_TYPE (AFU_IMAGE_MMIO_CMD_TYPE * 4)
#define MMIO_CMD_ARG0 (AFU_IMAGE_MMIO_CMD_ARG0 * 4)
#define MMIO_CMD_ARG1 (AFU_IMAGE_MMIO_CMD_ARG1 * 4)
#define MMIO_CMD_ARG2 (AFU_IMAGE_MMIO_CMD_ARG2 * 4)
#define MMIO_STATUS (AFU_IMAGE_MMIO_STATUS * 4)
#define MMIO_DEV_CAPS (AFU_IMAGE_MMIO_DEV_CAPS * 4)
#define MMIO_ISA_CAPS (AFU_IMAGE_MMIO_ISA_CAPS * 4)
#define MMIO_SCOPE_READ (AFU_IMAGE_MMIO_SCOPE_READ * 4)
#define MMIO_SCOPE_WRITE (AFU_IMAGE_MMIO_SCOPE_WRITE * 4)

using namespace vortex;

/* -------------------------------------------------------------------------- */
/* Helper structs                                                             */
/* -------------------------------------------------------------------------- */

struct DeviceRegs {
  uint64_t cmd_arg0 = 0; // staging addr (host)  or DCR addr
  uint64_t cmd_arg1 = 0; // device addr          or DCR data
  uint64_t cmd_arg2 = 0; // size (bytes)
  uint64_t status = 0;   // bit0 = BUSY, bits[63:8] = console (optional)
};

struct HostBuffer {
  uint8_t *data = nullptr;
  uint64_t size = 0;
  uint64_t ioaddr = 0;
};

/* -------------------------------------------------------------------------- */
/*  opae_simx::Impl                                                           */
/* -------------------------------------------------------------------------- */

class opae_simx::Impl {
public:
  Impl()
      : ram_(0, RAM_PAGE_SIZE), arch_(NUM_THREADS, NUM_WARPS, NUM_CORES), processor_(arch_), pinned_alloc_(0, PINNED_MEM_SIZE, RAM_PAGE_SIZE, CACHE_BLOCK_SIZE) {
    processor_.attach_ram(&ram_);
  }

  ~Impl() {
    if (run_future_.valid())
      run_future_.wait();
    for (auto &kv : host_buffers_)
      aligned_free(kv.second.data);
  }

  /* ---------- OPAE shim API --------------------------------------------- */

  int prepare_buffer(uint64_t len, void **buf_addr,
                     uint64_t *wsid, int /*flags*/) {
    auto mem = aligned_malloc(len, CACHE_BLOCK_SIZE);
    if (!mem)
      return -1;

    uint64_t pin_off;
    if (0 != pinned_alloc_.allocate(len, &pin_off))
      return -1;

    uint64_t id = next_buf_id_++;
    host_buffers_[id] = {reinterpret_cast<uint8_t *>(mem), len,
                         PIN_BASE_ADDR + pin_off};

    *buf_addr = mem; // host pointer (for memcpy in runtime)
    *wsid = id;
    return 0;
  }

  void release_buffer(uint64_t wsid) {
    auto it = host_buffers_.find(wsid);
    if (it == host_buffers_.end())
      return;
    pinned_alloc_.release(it->second.ioaddr - PIN_BASE_ADDR);
    aligned_free(it->second.data);
    host_buffers_.erase(it);
  }

  void get_io_address(uint64_t wsid, uint64_t *ioaddr) {
    *ioaddr = host_buffers_[wsid].ioaddr;
  }

  void write_mmio64(uint32_t /*mmio*/, uint64_t off, uint64_t val) {
    std::scoped_lock lk(mtx_);
    switch (off) {
    case MMIO_CMD_ARG0:
      regs_.cmd_arg0 = val;
      break;
    case MMIO_CMD_ARG1:
      regs_.cmd_arg1 = val;
      break;
    case MMIO_CMD_ARG2:
      regs_.cmd_arg2 = val;
      break;

    case MMIO_CMD_TYPE:
      if (val == AFU_IMAGE_CMD_MEM_WRITE)
        cmd_mem_write();
      else if (val == AFU_IMAGE_CMD_MEM_READ)
        cmd_mem_read();
      else if (val == AFU_IMAGE_CMD_RUN)
        cmd_run();
      else if (val == AFU_IMAGE_CMD_DCR_WRITE)
        cmd_dcr_write();
      break;
    default: /* ignore */;
    }
  }

  void read_mmio64(uint32_t /*mmio*/, uint64_t off, uint64_t *val) {
    std::scoped_lock lk(mtx_);
    *val = (off == MMIO_STATUS) ? regs_.status : 0;
  }

  /* ---------- end API --------------------------------------------------- */

private:
  /* ------ command helpers ---------------------------------------------- */

  void cmd_mem_write() {
    regs_.status = 1;
    memcpy(&ram_[regs_.cmd_arg1], host_ptr(regs_.cmd_arg0), regs_.cmd_arg2);
    regs_.status = 0;
  }

  void cmd_mem_read() {
    regs_.status = 1;
    memcpy(host_ptr(regs_.cmd_arg0), &ram_[regs_.cmd_arg1], regs_.cmd_arg2);
    regs_.status = 0;
  }

  void cmd_dcr_write() {
    uint32_t addr = static_cast<uint32_t>(regs_.cmd_arg0);
    uint32_t value = static_cast<uint32_t>(regs_.cmd_arg1);
    processor_.dcr_write(addr, value);
  }

  void cmd_run() {
    if (run_future_.valid())
      run_future_.wait();

    uint64_t krnl = regs_.cmd_arg0;
    uint64_t args = regs_.cmd_arg1;

    regs_.status = 1; // BUSY
    run_future_ = std::async(std::launch::async, [this, krnl, args] {
      processor_.dcr_write(VX_DCR_BASE_STARTUP_ADDR0, krnl & 0xffffffff);
      processor_.dcr_write(VX_DCR_BASE_STARTUP_ADDR1, krnl >> 32);
      processor_.dcr_write(VX_DCR_BASE_STARTUP_ARG0, args & 0xffffffff);
      processor_.dcr_write(VX_DCR_BASE_STARTUP_ARG1, args >> 32);
      processor_.run(); // run to completion
      regs_.status = 0; // clear BUSY
    });
  }

  /* ------ helpers ------------------------------------------------------- */

  uint8_t *host_ptr(uint64_t ioaddr) {
    return reinterpret_cast<uint8_t *>(ioaddr - PIN_BASE_ADDR + reinterpret_cast<uintptr_t>(host_buffers_.begin()->second.data));
  }

  /* ------ data members -------------------------------------------------- */

  RAM ram_;
  Arch arch_;
  Processor processor_;
  MemoryAllocator pinned_alloc_;

  std::unordered_map<uint64_t, HostBuffer> host_buffers_;
  uint64_t next_buf_id_ = 0;

  DeviceRegs regs_{};
  std::future<void> run_future_;
  std::mutex mtx_;
};

/* ----------------------------------------------------------------------- */
/*  Public opae_simx thin wrappers                                         */
/* ----------------------------------------------------------------------- */

opae_simx::opae_simx() : impl_(new Impl()) {}
opae_simx::~opae_simx() { delete impl_; }

int opae_simx::init() { return 0; }
void opae_simx::shutdown() {}

int opae_simx::prepare_buffer(uint64_t l, void **p, uint64_t *w, int f) { return impl_->prepare_buffer(l, p, w, f); }
void opae_simx::release_buffer(uint64_t w) { impl_->release_buffer(w); }
void opae_simx::get_io_address(uint64_t w, uint64_t *i) { impl_->get_io_address(w, i); }

void opae_simx::write_mmio64(uint32_t m, uint64_t o, uint64_t v) { impl_->write_mmio64(m, o, v); }
void opae_simx::read_mmio64(uint32_t m, uint64_t o, uint64_t *v) { impl_->read_mmio64(m, o, v); }
