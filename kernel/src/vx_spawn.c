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

#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include <vx_print.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

__thread dim3_t blockIdx;
__thread dim3_t threadIdx;
dim3_t gridDim;
dim3_t blockDim;

__thread uint32_t __local_group_id;
uint32_t __warps_per_group;

typedef struct {
	vx_kernel_func_cb callback;
	const void* arg;
	uint32_t group_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
  uint32_t warps_per_group;
  uint32_t groups_per_core;
  uint32_t remaining_mask;
} wspawn_groups_args_t;

typedef struct {
	vx_kernel_func_cb callback;
	const void* arg;
	uint32_t all_tasks_offset;
  uint32_t remain_tasks_offset;
	uint32_t warp_batches;
	uint32_t remaining_warps;
} wspawn_threads_args_t;

typedef struct
{
  vx_kernel_func_cb callback;
  const void *arg;
  uint32_t start_block_x;
  uint32_t start_block_y;
  uint32_t end_block_x;
  uint32_t end_block_y;
  uint32_t warp_batches;
  uint32_t remaining_warps;
  uint32_t warps_per_group;
  uint32_t remaining_mask;
  uint32_t group_size;
  uint32_t groups_at_once;
} wspawn_spatial_args_t;

// Function to calculate the integer square root using Newton-Raphson method
uint32_t integer_sqrt(uint32_t S)
{
  if (S == 0)
    return 0;
  uint32_t x = S;
  uint32_t y = (x + 1) / 2;
  while (y < x)
  {
    x = y;
    y = (x + S / x) / 2;
  }
  return x;
}

static void __attribute__ ((noinline)) process_threads() {
  wspawn_threads_args_t* targs = (wspawn_threads_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();

  uint32_t start_warp = (warp_id * targs->warp_batches) + MIN(warp_id, targs->remaining_warps);
  uint32_t iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  uint32_t start_task_id = targs->all_tasks_offset + (start_warp * threads_per_warp) + thread_id;
  uint32_t end_task_id = start_task_id + iterations * threads_per_warp;

  __local_group_id = 0;
  threadIdx.x = 0;
  threadIdx.y = 0;
  threadIdx.z = 0;

  vx_kernel_func_cb callback = targs->callback;
  const void* arg = targs->arg;

  for (uint32_t task_id = start_task_id; task_id < end_task_id; task_id += threads_per_warp) {
    blockIdx.x = task_id % gridDim.x;
    blockIdx.y = (task_id / gridDim.x) % gridDim.y;
    blockIdx.z = task_id / (gridDim.x * gridDim.y);
    callback((void*)arg);
  }
}

static void __attribute__ ((noinline)) process_remaining_threads() {
  wspawn_threads_args_t* targs = (wspawn_threads_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t thread_id = vx_thread_id();
  uint32_t task_id = targs->remain_tasks_offset + thread_id;

  (targs->callback)((void*)targs->arg);
}

static void __attribute__ ((noinline)) process_threads_stub() {
  // activate all threads
  vx_tmc(-1);

  // process all tasks
  process_threads();

  // disable warp
  vx_tmc_zero();
}

static void __attribute__ ((noinline)) process_thread_groups() {
  wspawn_groups_args_t* targs = (wspawn_groups_args_t*)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();

  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t groups_per_core = targs->groups_per_core;

  uint32_t iterations = targs->warp_batches + (warp_id < targs->remaining_warps);

  uint32_t local_group_id = warp_id / warps_per_group;
  uint32_t group_warp_id = warp_id - local_group_id * warps_per_group;
  uint32_t local_task_id = group_warp_id * threads_per_warp + thread_id;

  uint32_t start_group = targs->group_offset + local_group_id;
  uint32_t end_group = start_group + iterations * groups_per_core;

  __local_group_id = local_group_id;

  threadIdx.x = local_task_id % blockDim.x;
  threadIdx.y = (local_task_id / blockDim.x) % blockDim.y;
  threadIdx.z = local_task_id / (blockDim.x * blockDim.y);

  vx_kernel_func_cb callback = targs->callback;
  const void* arg = targs->arg;

  for (uint32_t group_id = start_group; group_id < end_group; group_id += groups_per_core) {
    blockIdx.x = group_id % gridDim.x;
    blockIdx.y = (group_id / gridDim.x) % gridDim.y;
    blockIdx.z = group_id / (gridDim.x * gridDim.y);
    callback((void*)arg);
  }
}

static void __attribute__((noinline)) process_thread_groups_spatial()
{
  wspawn_spatial_args_t *targs = (wspawn_spatial_args_t *)csr_read(VX_CSR_MSCRATCH);

  uint32_t threads_per_warp = vx_num_threads();
  uint32_t warp_id = vx_warp_id();
  uint32_t thread_id = vx_thread_id();
  uint32_t core_id = vx_core_id(); // Get the core ID
  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t group_size = targs->group_size;
  uint32_t groups_at_once = targs->groups_at_once;


  vx_kernel_func_cb callback = targs->callback;
  const void *arg = targs->arg;


  // Distribute blockIdx.z among global warps
  uint32_t total_blocks_z = gridDim.z;


  // **Calculate local_group_id and group_warp_id**
  uint32_t local_group_id = warp_id / warps_per_group;
  uint32_t group_warp_id = warp_id - local_group_id * warps_per_group; // same as warp_id % warps_per_group

  // Set local task ID within the group
  uint32_t local_task_id = group_warp_id * threads_per_warp + thread_id;

  // If this warp does not have valid tasks, return early
  if (local_task_id >= group_size)
  {
    return;
  }

  // Calculate thread indices
  threadIdx.x = local_task_id % blockDim.x;
  threadIdx.y = (local_task_id / blockDim.x) % blockDim.y;
  threadIdx.z = local_task_id / (blockDim.x * blockDim.y);

  for (uint32_t block_z = 0; block_z < total_blocks_z; block_z += groups_at_once)
  {
    for (uint32_t block_y = targs->start_block_y; block_y < targs->end_block_y; ++block_y)
    {
      for (uint32_t block_x = targs->start_block_x; block_x < targs->end_block_x; ++block_x)
      {
        blockIdx.x = block_x;
        blockIdx.y = block_y;
        blockIdx.z = block_z + local_group_id;

        // **Add Debugging Statements Here**
        //vx_printf("Core %d, Global Warp %d, Warp %d, Batch %d, Thread %d:\n", core_id, global_warp_id, warp_id, batch, thread_id);
        //vx_printf("  Processing BlockIdx: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
        //vx_printf("  ThreadIdx: (%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);

        // Call kernel function
        callback((void *)arg);
      }
    }
  }
}

static void __attribute__ ((noinline)) process_thread_groups_stub() {
  wspawn_groups_args_t* targs = (wspawn_groups_args_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t remaining_mask = targs->remaining_mask;
  uint32_t warp_id = vx_warp_id();
  uint32_t group_warp_id = warp_id % warps_per_group;
  uint32_t threads_mask = (group_warp_id == warps_per_group-1) ? remaining_mask : -1;

  // activate threads
  vx_tmc(threads_mask);

  // process thread groups
  process_thread_groups();

  // disable all warps except warp0
  vx_tmc(0 == vx_warp_id());
}

static void __attribute__((noinline)) process_thread_groups_spatial_stub()
{
  wspawn_spatial_args_t *targs = (wspawn_spatial_args_t *)csr_read(VX_CSR_MSCRATCH);
  uint32_t warps_per_group = targs->warps_per_group;
  uint32_t remaining_mask = targs->remaining_mask;
  uint32_t warp_id = vx_warp_id();
  uint32_t group_warp_id = warp_id % warps_per_group;
  uint32_t threads_mask = (group_warp_id == warps_per_group - 1) ? remaining_mask : -1;

  // activate threads
  vx_tmc(threads_mask);

  // process thread groups
  process_thread_groups_spatial();

  // disable all warps except warp0
  vx_tmc(0 == vx_warp_id());
}

int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t * block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg) {
  // calculate number of groups and group size
  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < 3; ++i) {
    uint32_t gd = (grid_dim && (i < dimension)) ? grid_dim[i] : 1;
    uint32_t bd = (block_dim && (i < dimension)) ? block_dim[i] : 1;
    num_groups *= gd;
    group_size *= bd;
    gridDim.m[i] = gd;
    blockDim.m[i] = bd;
  }

  // device specifications
  uint32_t num_cores = vx_num_cores();
  uint32_t warps_per_core = vx_num_warps();
  uint32_t threads_per_warp = vx_num_threads();
  uint32_t core_id = vx_core_id();

  // check group size
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (threads_per_core < group_size) {
    vx_printf("error: group_size > threads_per_core (%d,%d)\n", group_size, threads_per_core);
    return -1;
  }

  if (group_size > 1) {
    // calculate number of warps per group
    uint32_t warps_per_group = group_size / threads_per_warp;
    uint32_t remaining_threads = group_size - warps_per_group * threads_per_warp;
    uint32_t remaining_mask = -1;
    if (remaining_threads != 0) {
      remaining_mask = (1 << remaining_threads) - 1;
      ++warps_per_group;
    }

    // calculate necessary active cores
    uint32_t needed_warps = num_groups * warps_per_group;
    uint32_t needed_cores = (needed_warps + warps_per_core-1) / warps_per_core;
    uint32_t active_cores = MIN(needed_cores, num_cores);

    // only active cores participate
    if (core_id >= active_cores)
      return 0;

    // total number of groups per core
    uint32_t total_groups_per_core = num_groups / active_cores;
    uint32_t remaining_groups_per_core = num_groups - active_cores * total_groups_per_core;
    if (core_id < remaining_groups_per_core)
      ++total_groups_per_core;

    // calculate number of warps to activate
    uint32_t groups_per_core = warps_per_core / warps_per_group;
    uint32_t total_warps_per_core = total_groups_per_core * warps_per_group;
    uint32_t active_warps = total_warps_per_core;
    uint32_t warp_batches = 1, remaining_warps = 0;
    if (active_warps > warps_per_core) {
      active_warps = groups_per_core * warps_per_group;
      warp_batches = total_warps_per_core / active_warps;
      remaining_warps = total_warps_per_core - warp_batches * active_warps;
    }

    // calculate offsets for group distribution
    uint32_t group_offset = core_id * total_groups_per_core + MIN(core_id, remaining_groups_per_core);

    // set scheduler arguments
    wspawn_groups_args_t wspawn_args = {
      kernel_func,
      arg,
      group_offset,
      warp_batches,
      remaining_warps,
      warps_per_group,
      groups_per_core,
      remaining_mask
    };
    csr_write(VX_CSR_MSCRATCH, &wspawn_args);

    // set global variables
    __warps_per_group = warps_per_group;

    // execute callback on other warps
    vx_wspawn(active_warps, process_thread_groups_stub);

    // execute callback on warp0
    process_thread_groups_stub();
  } else {
    uint32_t num_tasks = num_groups;
    __warps_per_group = 0;

    // calculate necessary active cores
    uint32_t needed_cores = (num_tasks + threads_per_core - 1) / threads_per_core;
    uint32_t active_cores = MIN(needed_cores, num_cores);

    // only active cores participate
    if (core_id >= active_cores)
      return 0;

    // number of tasks per core
    uint32_t tasks_per_core = num_tasks / active_cores;
    uint32_t remaining_tasks_per_core = num_tasks - tasks_per_core * active_cores;
    if (core_id < remaining_tasks_per_core)
      ++tasks_per_core;

    // calculate number of warps to activate
    uint32_t total_warps_per_core = tasks_per_core / threads_per_warp;
    uint32_t remaining_tasks = tasks_per_core - total_warps_per_core * threads_per_warp;
    uint32_t active_warps = total_warps_per_core;
    uint32_t warp_batches = 1, remaining_warps = 0;
    if (active_warps > warps_per_core) {
      active_warps = warps_per_core;
      warp_batches = total_warps_per_core / active_warps;
      remaining_warps = total_warps_per_core - warp_batches * active_warps;
    }

    // calculate offsets for task distribution
    uint32_t all_tasks_offset = core_id * tasks_per_core + MIN(core_id, remaining_tasks_per_core);
    uint32_t remain_tasks_offset = all_tasks_offset + (tasks_per_core - remaining_tasks);

    // prepare scheduler arguments
    wspawn_threads_args_t wspawn_args = {
      kernel_func,
      arg,
      all_tasks_offset,
      remain_tasks_offset,
      warp_batches,
      remaining_warps
    };
    csr_write(VX_CSR_MSCRATCH, &wspawn_args);

    if (active_warps >= 1) {
      // execute callback on other warps
      vx_wspawn(active_warps, process_threads_stub);

      // activate all threads
      vx_tmc(-1);

      // process threads
      process_threads();

      // back to single-threaded
      vx_tmc_one();
    }

    if (remaining_tasks != 0) {
      // activate remaining threads
      uint32_t tmask = (1 << remaining_tasks) - 1;
      vx_tmc(tmask);

      // process remaining threads
      process_remaining_threads();

      // back to single-threaded
      vx_tmc_one();
    }
  }

  // wait for spawned warps to complete
  vx_wspawn(1, 0);

  return 0;
}

// spatial spawn
int vx_spawn_threads_spatial(uint32_t dimension,
                     const uint32_t *grid_dim,
                     const uint32_t *block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void *arg)
{
  vx_printf("SPATIAL MODE\n");
  // calculate number of groups and group size
  uint32_t num_groups = 1;
  uint32_t group_size = 1;
  for (uint32_t i = 0; i < 3; ++i)
  {
    uint32_t gd = (grid_dim && (i < dimension)) ? grid_dim[i] : 1;
    uint32_t bd = (block_dim && (i < dimension)) ? block_dim[i] : 1;
    num_groups *= gd;
    group_size *= bd;
    gridDim.m[i] = gd;
    blockDim.m[i] = bd;
  }

  // device specifications
  uint32_t num_cores = vx_num_cores();
  uint32_t warps_per_core = vx_num_warps();
  uint32_t threads_per_warp = vx_num_threads();
  uint32_t core_id = vx_core_id();

  // check group size
  uint32_t threads_per_core = warps_per_core * threads_per_warp;
  if (threads_per_core < group_size)
  {
    vx_printf("error: group_size > threads_per_core (%d,%d)\n", group_size, threads_per_core);
    return -1;
  }

  // calculate number of cores per dimension in x y plane
  uint32_t core_grid_dim = integer_sqrt(num_cores);
  if (core_grid_dim * core_grid_dim != num_cores)
  {
    vx_printf("error: num_cores must have an whole number sqrt: %d\n", num_cores);
    return -1;
  }

  // calculate number of warps per group
  uint32_t warps_per_group = group_size / threads_per_warp;
  uint32_t remaining_threads = group_size - warps_per_group * threads_per_warp;
  uint32_t remaining_mask = -1;
  if (remaining_threads != 0)
  {
    vx_printf("error: group_size must be a multipule of threads_per_warp (%d,%d)\n", group_size, threads_per_warp);
    return -1;
  }

  uint32_t groups_at_once = threads_per_core / group_size;
  if (threads_per_core != groups_at_once * group_size)
  {
    vx_printf("error: threads_per_core must be a multipule of group_size (%d,%d)\n", threads_per_core, group_size);
  }

  // calculate necessary active cores based on grid dimensions
  uint32_t active_cores_x = MIN(gridDim.x, core_grid_dim);
  uint32_t active_cores_y = MIN(gridDim.y, core_grid_dim);

  vx_printf("active_cores_x: %d, active_cores_y: %d\n", active_cores_x, active_cores_y);
  // check if the current core is active
  uint32_t core_x = core_id % core_grid_dim;
  uint32_t core_y = core_id / core_grid_dim;
  if (core_x >= active_cores_x || core_y >= active_cores_y)
  {
    return 0;
  }
  vx_printf("core id %d is active\n", core_id);
  uint32_t total_groups_x = gridDim.x / active_cores_x;
  uint32_t total_groups_y = gridDim.y / active_cores_y;
  uint32_t remaining_groups_x = gridDim.x % active_cores_x;
  uint32_t remaining_groups_y = gridDim.y % active_cores_y;

  uint32_t start_block_x = core_x * total_groups_x + MIN(core_x, remaining_groups_x);
  uint32_t start_block_y = core_y * total_groups_y + MIN(core_y, remaining_groups_y);
  uint32_t end_block_x = start_block_x + total_groups_x + (core_x < remaining_groups_x);
  uint32_t end_block_y = start_block_y + total_groups_y + (core_y < remaining_groups_y);

  // total number of groups per core
  uint32_t total_groups_per_core = (end_block_x - start_block_x) * (end_block_y - start_block_y) * gridDim.z;
  vx_printf("total_groups_per_core: %d\n", total_groups_per_core);
  // Calculate the number of warps to activate
  uint32_t total_warps_per_core = total_groups_per_core * warps_per_group;
  uint32_t active_warps = (group_size + threads_per_warp - 1) / threads_per_warp; // Round up
  if(groups_at_once > 1)
  {
    active_warps = groups_at_once * warps_per_group;
  }
  uint32_t warp_batches = 1;
  uint32_t remaining_warps = 0;

  // print out all wspawn_args
  vx_printf("kernel_func: %p\n", kernel_func);
  vx_printf("arg: %p\n", arg);
  vx_printf("core %d: start_block_x: %d, end_block_x: %d, start_block_y: %d, end_block_y: %d \n", core_id, start_block_x, end_block_x, start_block_y, end_block_y);
  vx_printf("warp_batches: %d\n", warp_batches);
  vx_printf("remaining_warps: %d\n", remaining_warps);
  vx_printf("warps_per_group: %d\n", warps_per_group);
  vx_printf("remaining_mask: %d\n", remaining_mask);
  vx_printf("active_warps: %d\n", active_warps);
  vx_printf("groups_at_once: %d\n", groups_at_once);
  


  // Set scheduler arguments
  wspawn_spatial_args_t wspawn_args = {
      .callback = kernel_func,
      .arg = arg,
      .start_block_x = start_block_x,
      .start_block_y = start_block_y,
      .end_block_x = end_block_x,
      .end_block_y = end_block_y,
      .warp_batches = warp_batches,
      .remaining_warps = remaining_warps,
      .warps_per_group = warps_per_group,
      .remaining_mask = remaining_mask,
      .group_size = group_size,
      .groups_at_once = groups_at_once};
  csr_write(VX_CSR_MSCRATCH, &wspawn_args);

  // set global variables
  __warps_per_group = warps_per_group;

  // execute callback on other warps
  vx_wspawn(active_warps, process_thread_groups_spatial_stub);

  // execute callback on warp0
  process_thread_groups_spatial_stub();



// wait for spawned warps to complete
vx_wspawn(1, 0);

return 0;
}

#ifdef __cplusplus
}
#endif
