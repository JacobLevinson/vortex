
// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensor_unit.h"
#include "core.h"

using namespace vortex;

class TensorUnit::Impl {
public:
  Impl(TensorUnit* simobject, const Arch& arch, Core* core)
    : simobject_(simobject)
    , core_(core)
    , arch_(arch)
    , perf_stats_()
  {
    //--
  }

  ~Impl() {
    // Destructor logic if needed
  }

  void reset() {
    perf_stats_ = PerfStats();
  }

  void tick() {
    for (uint32_t iw = 0; iw < ISSUE_WIDTH; ++iw) {
      auto& input = simobject_->Inputs.at(iw);
      if (input.empty())
          return;
      auto trace = input.front();
      int delay = 0;
      switch (trace->tpu_type) {
      case TpuType::HMMA844:
        delay = 4;
        break;
      default:
        std::abort();
      }
      simobject_->Outputs.at(iw).push(trace, 2 + delay);
      DT(3, simobject_->name() << ": op=" << trace->tpu_type << ", " << *trace);
      input.pop();
    }
  }

  void hmma844(uint32_t wid,
                                 uint32_t fmt,
                                 uint32_t step,
                                 const std::vector<reg_data_t> &rs1_data,
                                 const std::vector<reg_data_t> &rs2_data,
                                 const std::vector<reg_data_t> &rs3_data,
                                 std::vector<reg_data_t> &rd_data,
                                 ExeTraceData *trace_data) {
    // 1) pull the full 32 lanes of each VReg into temporaries
    float va_data[32], vb_data[32], vc_data[32];
    for (int lane = 0; lane < 32; lane++) {
      va_data[lane] = rs1_data[lane].f32; // 8×4 block of A
      vb_data[lane] = rs2_data[lane].f32; // 2×(4×4) blocks of B
      vc_data[lane] = rs3_data[lane].f32; // 8×4 block of C (or prior D)
    }

    // 2) pick which 4×4 sub-tile of B to use
    int cb = step & 3;   // bits [1:0]
    int half = cb & 1;   // bit 0 → choose low or high half
    int off = half * 16; // each half is 4×4 = 16 floats

    // 3) extract subB[4][4]
    float subB[4][4];
    for (int x = 0; x < 4; x++) {
      for (int y = 0; y < 4; y++) {
        subB[x][y] = vb_data[off + x * 4 + y];
      }
    }

    // 4) for each lane (32 lanes = 8×4 tile) do the dot-product + add
    for (int lane = 0; lane < 32; lane++) {
      int x = lane / 4; // row in 8×4
      int y = lane % 4; // col in 8×4

      float sum = 0.0f;
      for (int z = 0; z < 4; z++) {
        // subA[x][z]  == va_data[x*4 + z]
        // subB[z][y]
        sum += va_data[x * 4 + z] * subB[z][y];
      }

      // acc = vc_data[lane]
      rd_data[lane].f32 = vc_data[lane] + sum;
    }
  }

  const PerfStats& perf_stats() const {
    return perf_stats_;
  }

private:
  TensorUnit*   simobject_;
  Core*         core_;
  Arch          arch_;
  PerfStats     perf_stats_;
};

///////////////////////////////////////////////////////////////////////////////

TensorUnit::TensorUnit(const SimContext &ctx, const char* name, const Arch& arch, Core* core)
	: SimObject<TensorUnit>(ctx, name)
	, Inputs(ISSUE_WIDTH, this)
	, Outputs(ISSUE_WIDTH, this)
	, impl_(new Impl(this, arch, core))
{}

TensorUnit::~TensorUnit() {
  delete impl_;
}

void TensorUnit::reset() {
  impl_->reset();
}

void TensorUnit::tick() {
  impl_->tick();
}

const TensorUnit::PerfStats &TensorUnit::perf_stats() const {
	return impl_->perf_stats();
}

void TensorUnit::hmma844(uint32_t wid,
                         uint32_t fmt, uint32_t step,
                         const std::vector<reg_data_t>& rs1_data,
                         const std::vector<reg_data_t>& rs2_data,
                         const std::vector<reg_data_t>& rs3_data,
                         std::vector<reg_data_t>& rd_data,
                         ExeTraceData* trace_data) {
  impl_->hmma844(wid, fmt, step, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
}