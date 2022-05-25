//
// Created by Djordje on 5/24/2022.
//

#ifndef PVDB_TEST_GPU_UTILS_H
#define PVDB_TEST_GPU_UTILS_H

#include "catch.hpp"

#include "../gpu/DebugDevice.h"
#include "../gpu/Context.h"
#include "../gpu/Pass.h"

struct ContextDestroy {
    pvdb::gpu::DebugDevice& device;
    pvdb::gpu::Context& context;
    ~ContextDestroy() {
        device.wait_idle();
        context.destroy();
        device.destroy();
    }
};

pvdb::gpu::Context& get_context()
{
    static bool is_init = false;
    static pvdb::gpu::DebugDevice g_device{};
    static pvdb::gpu::Context g_ctx{};
    static ContextDestroy destroy{g_device, g_ctx};
    if (!is_init) {
        g_device.init();
        g_ctx.init(g_device.context_desc());
        is_init = true;
    }
    return g_ctx;
}

#endif //PVDB_TEST_GPU_UTILS_H
