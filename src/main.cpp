//
// Created by Djordje on 5/21/2022.
//

#include "gpu/Context.h"
#include "gpu/DebugDevice.h"

/*
#define PVDB_C
#define PVDB_64_BIT
#include "pvdb/pvdb_buffer.h"
#include "pvdb/pvdb_math.h"
#include "pvdb/pvdb_tree_read.h"
#include "pvdb/pvdb_tree_write.h"

int main() {
    auto& nodes = *new pvdb_buf_t<100000>{};
    auto& atlas = *new pvdb_buf_t<1000000>{};
    pvdb_buf_t<256> alloc{};
    pvdb_tree tree{};
    pvdb_tree_init(tree, 4, {1, 1, 1, 1}, 0u);
    pvdb_allocator_init(alloc, 3u, {
        {1u, 8u, 1u},
        {pvdb_node_size(tree, 1), 8u, -1u},
        {pvdb_node_size(tree, 2), 8u, -1u},
    }, pvdb_node_size(tree, pvdb_root(tree)));
    tree.data.nodes = nodes.data;
    tree.data.alloc = alloc.data;
    tree.data.atlas = atlas.data;

//    uint allocated{}, level{pvdb_root(tree)};
//    uint path[PVDB_MAX_LEVELS]{};
//    for (;;) {
//        path[level] = level == 0u ? 1u : allocated;
//        allocated += pvdb_node_size(tree, level);
//        if (level-- == 0) break;
//    }
//    level = pvdb_root(tree);
//    for (;;) {
//        pvdb_write_node_mask(tree, path[level], 0, true);
//        pvdb_write_node(tree, 0u, path[level], 0u, level == 0u ? 1u : path[level - 1]);
//        if (level-- == 1) break;
//    }
//    pvdb_write_leaf(tree, 1u, {0, 0, 0}, 999u);

    pvdb_set(tree, {7,7,7}, 999u);

    printf("%u\n", pvdb_get(tree, {7,7,7}));
}
*/

#include "cvdb/Tree.h"
#include "cvdb/Runtime.h"
#include "cvdb/Camera.h"
#include "cvdb/Debug.h"

int main() {
    pvdb::gpu::DebugDevice device{};
    device.init(1200, 800);

    pvdb::Runtime rt{};
    rt.init({device.device, device.vma, device.queues.compute, device.queues.graphics, device.queues.family_compute, device.render_pass.render_pass});

    pvdb::DebugController ct{};
    ct.cam.reset(80.0f, float(device.swapchain.width) / float(device.swapchain.height));
    ct.cam.set_position({-2.0f, 2.0f, -2.0f});
    ct.cam.look_at({1.0f, 1.0f, 1.0f});
    device.events.on_mouse_move = [&](int, int, int mx, int my) { ct.on_mouse_move(mx, my); };
    device.events.on_key = [&](int k, bool pressed, bool repeated) { if (!repeated) ct.on_key(device, k, pressed); };

    auto cmd = rt.context().begin_setup();
#ifdef PVDB_USE_IMAGES
    auto t = rt.trees().create(cmd, {100'000'000, 3, {3, 4, 5}, 18u});
#else
    auto t = rt.trees().create(cmd, {100'000'000, 3, {3, 4, 5}});
#endif
    rt.context().end_setup();
    rt.context().wait_setup();
    rt.debug_fill(t, {0, 0, 0}, {512, 1, 512}, {});

    device.swapchain.render.waits_for(rt.last_submit(), pvdb::gpu::PipelineStage::COLOR_OUTPUT);

    while (!device.should_quit()) {
        ct.update(1.0f);
        rt.camera().update(ct.cam.view, ct.cam.proj);
        rt.context().update();
        if (auto i = device.acquire_image(rt.context()); i != UINT32_MAX) {
            device.set_title("pos: (%.2f, %.2f, %.2f), dir: (%.4f, %.4f, %.4f), draw time ms: %lf, MRays/s: %.1f\n",
                ct.cam.position.x, ct.cam.position.y, ct.cam.position.z,
                ct.cam.forward.x, ct.cam.forward.y, ct.cam.forward.z,
                double(device.swapchain.draw_time)/1000000.0,
                double(device.swapchain.width * device.swapchain.height) / (double(device.swapchain.draw_time)/1000000000.0) / 1000000.0);
            rt.draw(device.cmd().vk, pvdb::Runtime::Draw::DEFAULT);
            device.present(rt.context(), i);
        }
    }

    device.wait_idle();
    rt.trees().destroy(t);
    rt.destroy();
    device.destroy();
    return 0;
}