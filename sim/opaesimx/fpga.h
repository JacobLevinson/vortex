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

#ifndef __FPGA_H__
#define __FPGA_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FPGA_OK = 0,        /**< Operation completed successfully */
  FPGA_INVALID_PARAM, /**< Invalid parameter supplied */
  FPGA_BUSY,          /**< Resource is busy */
  FPGA_EXCEPTION,     /**< An exception occurred */
  FPGA_NOT_FOUND,     /**< A required resource was not found */
  FPGA_NO_MEMORY,     /**< Not enough memory to complete operation */
  FPGA_NOT_SUPPORTED, /**< Requested operation is not supported */
  FPGA_NO_DRIVER,     /**< Driver is not loaded */
  FPGA_NO_DAEMON,     /**< FPGA Daemon (fpgad) is not running */
  FPGA_NO_ACCESS,     /**< Insufficient privileges or permissions */
  FPGA_RECONF_ERROR   /**< Error while reconfiguring FPGA */
} fpga_result;

typedef enum {
  FPGA_DEVICE = 0,
  FPGA_ACCELERATOR
} fpga_objtype;

typedef void *fpga_handle;

typedef void *fpga_token;

typedef void *fpga_properties;

typedef uint8_t fpga_guid[16];

// Function prototypes in single-line format
fpga_result fpgaGetProperties(fpga_token token, fpga_properties *prop);
fpga_result fpgaPropertiesSetObjectType(fpga_properties prop, fpga_objtype objtype);
fpga_result fpgaPropertiesSetGUID(fpga_properties prop, fpga_guid guid);
fpga_result fpgaDestroyProperties(fpga_properties *prop);
fpga_result fpgaEnumerate(const fpga_properties *filters, uint32_t num_filters, fpga_token *tokens, uint32_t max_tokens, uint32_t *num_matches);
fpga_result fpgaDestroyToken(fpga_token *token);
fpga_result fpgaPropertiesGetLocalMemorySize(const fpga_properties *filters, uint64_t *lms);
fpga_result fpgaOpen(fpga_token token, fpga_handle *handle, int flags);
fpga_result fpgaClose(fpga_handle handle);
fpga_result fpgaPrepareBuffer(fpga_handle handle, uint64_t len, void **buf_addr, uint64_t *wsid, int flags);
fpga_result fpgaReleaseBuffer(fpga_handle handle, uint64_t wsid);
fpga_result fpgaGetIOAddress(fpga_handle handle, uint64_t wsid, uint64_t *ioaddr);
fpga_result fpgaWriteMMIO64(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t value);
fpga_result fpgaReadMMIO64(fpga_handle handle, uint32_t mmio_num, uint64_t offset, uint64_t *value);
const char *fpgaErrStr(fpga_result e);

#ifdef __cplusplus
}
#endif

#endif // __FPGA_H__