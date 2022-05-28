//
// Created by Djordje on 5/24/2022.
//

#include "test_gpu_utils.h"
#include "catch.hpp"

#include "../cvdb/objects/Tree.h"
#include "../pvdb/pvdb_global_test.h"

using namespace pvdb::gpu;

static constexpr std::string_view SRC_SET = R"(#version 450
//#extension GL_EXT_debug_printf : enable
//#define PVDB_ENABLE_PRINTF

layout(std430, binding = 3, set = 0) buffer ResultBuffer { uint data[]; } result;

#include "pvdb/pvdb_tree.glsl"

layout(push_constant) uniform constants {
	uint count;
    uint step;
    uint cmd;
    pvdb_tree tree;
} PC;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main() {
    if (gl_GlobalInvocationID.x >= PC.count || gl_GlobalInvocationID.y >= PC.count || gl_GlobalInvocationID.z >= PC.count)
        return;
    const ivec3 p = ivec3(gl_GlobalInvocationID * PC.step);
    if (PC.cmd == 0) {
        pvdb_set(PC.tree, p, uint(p.x * 1000000 + p.y * 1000 + p.z));
    } else {
        const uint local_index = pvdb_coord_to_index(gl_LocalInvocationID, 3u);
        const uint global_index = gl_WorkGroupID.x * gl_NumWorkGroups.y * gl_NumWorkGroups.z
            + gl_WorkGroupID.y * gl_NumWorkGroups.z
            + gl_WorkGroupID.z;
        result.data[local_index + global_index * 512u] = pvdb_get(PC.tree, p);
    }
}
)";

struct TestSetData {
    Timer timer{};
    Buffer result{};
    pvdb::Trees trees{};
    uint ti{};
    Pipeline pipeline{};
    Submit submit{};

    void init(Context& ctx) {
        submit = ctx.create_submit(IsGraphics::NO);
        timer = Timer::create(ctx.device, ctx.vma, 2);
        result = ctx.create_buffer(256 * 256 * 256 * 4u, BufferType::STORAGE, BufferUsage::CPU_TO_GPU);
        ctx.bind(3, BufferType::STORAGE, {&result, 1}, 0);

        trees.init(ctx);
        ctx.setup();

        auto setup = ctx.begin_setup();
        ti = trees.create(setup, {100'000'000, 3, {3, 4, 5}}, pvdb::DEVICE_GPU);
        ctx.end_setup();
        ctx.wait_setup();

        for (uint i = 0; i < FRAMES_IN_FLIGHT; ++i)
            ctx.update();

        pipeline = ctx.create_compute({SRC_SET, {}});
    }

    void destroy(const Context& ctx) {
        submit.destroy();
        trees.destroy(ti);
        timer.destroy(ctx.device, ctx.vma);
        ctx.destroy_buffer(result);
    }

    uint64_t execute(const Context& ctx, uint count, uint step, bool read = false) {
        const uint n_exec = count / step;
        const uint w = n_exec < 8u ? 1u : n_exec / 8u;

        auto cmd = submit.begin(0);
        ctx.bind(cmd, IsGraphics::NO, 0);

        timer.reset(cmd);
        timer.record(cmd, IsGraphics::NO);

        cmd.bind_compute(pipeline);
        ctx.push_const(cmd, count, 0);
        ctx.push_const(cmd, step, 4);
        ctx.push_const(cmd, uint(read), 8);
        trees[ti].push_const(ctx, cmd, 12);
        cmd.dispatch(w, w, w);

        timer.record(cmd, IsGraphics::NO);
        timer.copy_results(cmd);

        submit.submit(ctx, 0, true);
        submit.wait(0);

        return timer.delta_time(0, 1);
    }
};


TEST_CASE("gpu_pvdb_set", "[pvdb]")
{
    const uint count = 256;
    const uint step = 4;

    auto& ctx = get_context();
    TestSetData test{};
    test.init(ctx);

    // write
    auto time_write = test.execute(ctx, count, step);
//    printf("Time: %llu ns\n", time_write);

    // read
    test.execute(ctx, count, step, true);

    // Verify
    static constexpr auto get_val = [](int x, int y, int z) {
        return uint(x * 1000000 + y * 1000 + z);
    };

    const uint n_exec = count / step;
    const uint w = n_exec < 8u ? 1u : n_exec / 8u;
    auto* data = (const uint*)test.result.data;
    for (int gx = 0; gx < w; ++gx) {
        for (int gy = 0; gy < w; ++gy) {
            for (int gz = 0; gz < w; ++gz) {
                for (int lx = 0; lx < 8; lx += step) {
                    for (int ly = 0; ly < 8; ly += step) {
                        for (int lz = 0; lz < 8; lz += step) {
                            const int x{(gx*8+lx)*int(step)}, y{(gy*8+ly)*int(step)}, z{(gz*8+lz)*int(step)};
                            if (x >= count || y >= count || z >= count) continue;
                            const uint global_index = gx * int(w) * int(w) + gy * int(w) + gz;
                            const uint local_index = pvdb_coord_to_index(ivec3(lx, ly, lz), 3u);
                            const uint r = data[global_index * 512 + local_index];
                            const uint expected = get_val(x,y,z);
                            REQUIRE( r == expected );
                        }
                    }
                }
            }
        }
    }

    test.destroy(ctx);
}

TEST_CASE("gpu_pvdb_set_bench", "[pvdb]")
{
    const uint64_t count = 64;
    const uint64_t step = 1;

    auto& ctx = get_context();
    TestSetData test{};
    test.init(ctx);

    // write
    auto time_write = test.execute(ctx, count, step);
    const uint64_t blocks_per_second = count * count * count / step / step / step * 1'000'000'000ull / time_write;
    auto cube_size_per_second = pow(double(blocks_per_second), 0.33333);
    auto square_size_per_second = pow(double(blocks_per_second), 0.5);
    printf("SET REGION: %llux%llux%llu -> \n\ttime: %lf ms \n\tMvoxel/sec: %lf"
           "\n\tcube/second: %lf \n\tsquare/second: %lf\n",
           count, count, count,
           double(time_write) / double(1000000.0), double(blocks_per_second)/1000000.0,
           cube_size_per_second, square_size_per_second);

    test.destroy(ctx);

    destroy_context(ctx);
}
