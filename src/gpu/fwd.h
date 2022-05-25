//
// Created by Djordje on 5/14/2022.
//

#ifndef VDB_GPU_FWD_H
#define VDB_GPU_FWD_H

#include "tinyvk_fwd.h"
#include "tinystd_span.h"
#include <cstdint>

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

}

namespace pvdb::gpu {

struct DebugDevice;
struct Context;
struct ContextDesc;
struct BufferView;
struct Buffer;
struct Image;
struct Cmd;
struct RenderPass;
struct Pipeline;
struct Timer;
struct Submit;
struct Pass;
struct Subpass;


enum {
    FRAMES_IN_FLIGHT = 2,
};

enum class GPUOnly { NO, YES };

enum class IsGraphics { NO, YES };

template<typename T>
struct PerFrame {
    T values[FRAMES_IN_FLIGHT];
    constexpr operator T*() { return &values[0]; }
    constexpr operator const T*() const { return &values[0]; }
    constexpr T& operator[](usize i) { return values[i]; }
    constexpr const T& operator[](usize i) const { return values[i]; }
};

}

#endif //VDB_GPU_FWD_H
