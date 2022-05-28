//
// Created by Djordje on 5/24/2022.
//

#ifndef PVDB_TEST_GPU_UTILS_H
#define PVDB_TEST_GPU_UTILS_H

#include "catch.hpp"

#include "../gpu/DebugDevice.h"
#include "../gpu/Context.h"
#include "../gpu/Pass.h"

#include <string_view>


inline pvdb::gpu::DebugDevice& get_device()
{
    static bool is_init = false;
    static pvdb::gpu::DebugDevice g_device{};
    static struct DeviceDestroy { pvdb::gpu::DebugDevice& d; ~DeviceDestroy() { d.destroy(); } } destroy{g_device};
    if (!is_init) {
        g_device.init();
        is_init = true;
    }
    return g_device;
}

inline pvdb::gpu::Context& get_context()
{
    static pvdb::gpu::Context g_ctx{};
    if (!g_ctx.device)
        g_ctx.init(get_device().context_desc());
    return g_ctx;
}

inline void destroy_context(pvdb::gpu::Context& ctx) {
    get_device().wait_idle();
    ctx.destroy();
    ctx.device = nullptr;
}


namespace pvdb::gpu {

struct GPUTest {
    Timer timer{};
    Buffer result{};
    Pipeline pipeline{};
    Submit submit{};

    struct Result {
        u32 size{};
        u32 binding{};
    };

    void init(Context& ctx, Result r, std::string_view src, std::initializer_list<ShaderMacro> macros = {}) {
        submit = ctx.create_submit(IsGraphics::NO);
        timer = Timer::create(ctx.device, ctx.vma, 2);
        result = ctx.create_buffer(r.size * sizeof(u32), BufferType::STORAGE, BufferUsage::CPU_TO_GPU);
        ctx.bind(r.binding, BufferType::STORAGE, {&result, 1}, 0);
        ctx.setup();

        auto setup = ctx.begin_setup();
        setup.fill_buffer(result, 0, result.size, 0u);
        ctx.end_setup();
        ctx.wait_setup();

        for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i)
            ctx.update();

        u32 nm = 0;
        ShaderMacro m[256]{};
        for (auto& macro: macros) m[nm++] = macro;
        m[nm++] = {"RESULT_BINDING", to_string(r.binding)};
        m[nm++] = {"GID", "gl_GlobalInvocationID"};
        m[nm++] = {"LID", "gl_LocalInvocationID"};
        m[nm++] = {"LSZ", "gl_WorkGroupSize"};
        m[nm++] = {"WID", "gl_WorkGroupID"};
        m[nm++] = {"LIDX", "gl_LocalInvocationIndex"};
        pipeline = ctx.create_compute({src, {m, nm}});
    }

    void destroy(const Context& ctx) {
        submit.destroy();
        timer.destroy(ctx.device, ctx.vma);
        ctx.destroy_buffer(result);
    }

    template<typename Exec>
    uint64_t execute(const Context& ctx, Exec&& exec) {
        auto cmd = submit.begin(0);
        ctx.bind(cmd, IsGraphics::NO, 0);

        timer.reset(cmd);
        timer.record(cmd, IsGraphics::NO);

        cmd.bind_compute(pipeline);
        exec(cmd);

        timer.record(cmd, IsGraphics::NO);
        timer.copy_results(cmd);

        submit.submit(ctx, 0, true);
        submit.wait(0);

        return timer.delta_time(0, 1);
    }
};

}

#endif //PVDB_TEST_GPU_UTILS_H
