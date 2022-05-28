//
// Created by Djordje on 5/26/2022.
//

#include "catch.hpp"
#include "test_gpu_utils.h"

#define PVDB_C
#include "../pvdb/pvdb_global_test.h"
PVDB_DEFINE_BUFFER_STORAGE(global_buffer);

using namespace pvdb;
using namespace pvdb::gpu;


static constexpr std::string_view SRC_GLOBAL = R"(#version 450
#extension GL_EXT_debug_printf : enable

#include "pvdb/pvdb_global_test.h"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main(){
    debugPrintfEXT("\n\tbuffer: %u\n", global_read(0u, 0u));
    debugPrintfEXT("\n\tconst : %u\n", global_log2dim(0u));
})";

TEST_CASE("GlobalTest", "[pvdb]")
{
    auto& ctx = get_context();
    GPUTest test{};
    test.init(ctx, {1u, 0u}, SRC_GLOBAL);

    test.execute(ctx, [](Cmd cmd){
        cmd.dispatch();
    });

    test.destroy(ctx);
    destroy_context(ctx);
}

/*
#include "../cvdb/objects/Queue.h"

using namespace pvdb;
using namespace pvdb::gpu;


static constexpr std::string_view SRC_QUEUE = R"(#version 450
layout(std430, binding = 0)              buffer Input        { uint data[]; } input_data;
layout(std430, binding = RESULT_BINDING) buffer ResultBuffer { uint data[]; } result_data;

#define PVDB_QUEUE_TYPE     ivec4
#define PVDB_QUEUE_NAME     pvdb_queue_ivec4
#define PVDB_QUEUE_BINDING  2u
#define PVDB_ARRAY_SIZE     2u
#include "pvdb/pvdb_queue.glsl"

void pp_step0()
{
    pvdb_queue_add(0u, ivec4(int(input_data.data[GID.x])), 512);
}

void pp_step1()
{
    ivec4 r = ivec4(0);
    if (!pvdb_queue_get(0u, GID.x, r)) return;
    pvdb_queue_add(1u, r*2, 512);
}

void pp_step2()
{
    ivec4 r = ivec4(0);
    if (!pvdb_queue_get(1u, GID.x, r)) return;
    result_data.data[GID.x] = uint(r.x);
}


layout(push_constant) uniform constants { uint cmd; } PC;
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    PP_FUNCTION_PIPELINE(PC.cmd, pp_step0, pp_step1, pp_step2);
})";

TEST_CASE("gpu_pvdb_queue", "[pvdb]")
{
    static constexpr uint N = 512;

    auto& ctx = get_context();

    auto input = ctx.create_buffer(N * sizeof(u32), BufferType::STORAGE, BufferUsage::GPU);
    ctx.bind(0, BufferType::STORAGE, {&input, 1});

    auto setup = ctx.begin_setup();

    uint in_data[N]{};
    for (u32 i = 0; i < N; ++i) in_data[i] = i + 1;
    setup.update_buffer(input, 0, span<const uint>{in_data});

    Queues queues{};
    queues.init(ctx, 2u, 2u);
    ctx.end_setup();

    pvdb::gpu::GPUTest test{};
    test.init(ctx, {N, 1u}, SRC_QUEUE, {});

    setup = ctx.begin_setup();
    queues.init<ivec4>(setup, 0u, N, DEVICE_GPU);
    queues.init<ivec4>(setup, 1u, N, DEVICE_GPU);
    ctx.end_setup();
    ctx.wait_setup();
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) ctx.update();

    test.execute(ctx, [&](pvdb::gpu::Cmd cmd){
        ctx.push_const(cmd, u32(0), 0u);
        cmd.dispatch(1, 1, 1);

        ctx.push_const(cmd, u32(1), 0u);
        queues.execute(cmd, 0u);

        ctx.push_const(cmd, u32(2), 0u);
        queues.execute(cmd, 1u);
    });

    bool success[N]{};
    auto* res = (const atom_t*)test.result.data;
    for (u32 i = 0; i < N; ++i) {
        const u32 r = res[i];
        success[r/2 - 1u] = true;
    }
    for (auto s: success) REQUIRE( s );

    queues.destroy(0u);
    queues.destroy(1u);
    ctx.destroy_buffer(input);

    test.destroy(ctx);

    destroy_context(ctx);
}



static constexpr std::string_view SRC_QUEUE_COPY = R"(#version 450
layout(std430, binding = RESULT_BINDING) buffer ResultBuffer { uint data[]; } result_data;

#define PVDB_QUEUE_TYPE     ivec4
#define PVDB_QUEUE_NAME     pvdb_queue_ivec4
#define PVDB_QUEUE_BINDING  1u
#define PVDB_ARRAY_SIZE     2u
#include "pvdb/pvdb_queue.glsl"

layout(push_constant) uniform constants { uint offset; } PC;
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
void main() {
    ivec4 r = ivec4(0);
    if (!pvdb_queue_get(1u, GID.x, r)) return;
    result_data.data[PC.offset + GID.x] = uint(r.x);
})";

TEST_CASE("gpu_pvdb_queue_copy", "[pvdb]")
{
    static constexpr uint N = 512;
    static constexpr uint N_CPU = 1024;

    auto& ctx = get_context();

    auto setup = ctx.begin_setup();

    Queues queues{};
    queues.init(ctx, 1u, 2u);
    ctx.end_setup();

    pvdb::gpu::GPUTest test{};
    test.init(ctx, {N_CPU, 0u}, SRC_QUEUE_COPY, {});

    setup = ctx.begin_setup();
    queues.init<ivec4>(setup, 0u, N_CPU, DEVICE_CPU);
    queues.init<ivec4>(setup, 1u, N, DEVICE_GPU);
    ctx.end_setup();
    ctx.wait_setup();
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) ctx.update();

    for (uint i = 0; i < N_CPU; ++i)
        queues.push(0u, ivec4(int(1u + i)));

    test.execute(ctx, [&](pvdb::gpu::Cmd cmd){
        u32 offset = 0;
        do {
            ctx.push_const(cmd, offset);
            offset = queues.fill(cmd, 1u, 0u, N, offset);
            queues.execute(cmd, 1u);
        } while(offset > 0u);
    });

    bool success[N_CPU]{};
    auto* res = (const atom_t*)test.result.data;
    for (u32 i = 0; i < N_CPU; ++i) {
        const u32 r = res[i];
        success[r - 1u] = true;
    }
    for (auto s: success) REQUIRE( s );

    queues.destroy(0u);
    queues.destroy(1u);

    test.destroy(ctx);

    destroy_context(ctx);
}
*/
