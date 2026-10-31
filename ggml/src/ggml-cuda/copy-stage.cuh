#pragma once

// Host -> Device copies through a pinned ring.
//
// The model's expert weights are mmap'ed. cudaHostRegister refuses a file mapping
// ("operation not supported", CUDA 13.4 on Linux, measured with tools/pinprobe.cu), so every copy
// of an expert leaves the driver's staging path: half the PCIe bandwidth (13,9 against 26,2 GB/s
// measured on an RTX 3090 Ti, Gen4 x16) and a submitting thread that blocks in the driver.
//
// The ring is anonymous pinned memory, which the driver does accept: the bytes are copied once
// into it (a plain memcpy, spread over a few threads) and the DMA then runs at the full rate.
//
// Both functions return true when the copy was queued, false when the caller has to copy itself
// (turned off, source already pinned, too small to be worth the extra copy, or no ring).
//
// Off by default: on the machine this was built for (RTX 3090 Ti, Gen4 x16, CUDA 13.4) the driver's
// own staging of an unregistered copy already reaches what the ring reaches, and the extra memcpy
// costs the submitting thread — 1.548 t/s prefill with eight fill threads against 1.558 without
// (docs/saettigung/03). GGML_CUDA_STAGE=1 turns it on; tools/pinprobe.cu measures a machine.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct cudaStream_t_st;

bool ggml_cuda_stage_h2d(int device, void * dst, const void * src, size_t bytes, void * stream);
bool ggml_cuda_stage_2d_h2d(int device, void * dst, size_t dpitch, const void * src, size_t spitch,
                            size_t width, size_t height, void * stream);

#ifdef __cplusplus
}
#endif
