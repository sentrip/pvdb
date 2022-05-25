//
// Created by Djordje on 5/22/2022.
//

#ifndef PVDB_CVDB_FWD_H
#define PVDB_CVDB_FWD_H

#define PVDB_C
#define PVDB_ALLOCATOR_MASK

#include <cstdint>
#include "tinystd_span.h"

namespace pvdb {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using f32 = float;
using f64 = double;
using usize = size_t;

using tinystd::span;

struct Allocator;
struct Camera;
struct Tree;
struct Trees;
struct Runtime;

enum Device {
    DEVICE_CPU,
    DEVICE_GPU,
};

}

namespace pvdb::gpu {

struct Context;
struct Cmd;
struct Buffer;
struct Image;
struct Submit;

}

namespace pvdb {

using gpu_context = const gpu::Context&;
using gpu_cmd = gpu::Cmd;


static constexpr u32 GPU_IMAGE_SIZE_BYTES = 32;
static constexpr u32 GPU_BUFFER_SIZE_BYTES = 40;
static constexpr u32 GPU_BUFFER_VIEW_SIZE_BYTES = 24;


template<usize N>
struct slots {
    slots() = default;

    bool empty()  const { return freelist_size == count; }
    u32  alloc()        { return freelist_size ? freelist[--freelist_size] : count++; }
    void free(u32 slot) { freelist[freelist_size++] = slot; }

private:
    u32 freelist[N]{};
    u32 freelist_size{};
    u32 count{};
};

}

#endif //PVDB_CVDB_FWD_H
