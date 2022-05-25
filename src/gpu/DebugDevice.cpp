//
// Created by Djordje on 5/14/2022.
//

#include "DebugDevice.h"
#include "Context.h"
#include <cstdarg>

#define TINYVK_USE_VMA
#include "tinyvk_device.h"
#include "tinyvk_queue.h"
#include "tinyvk_swapchain.h"
#include "tinyvk_renderpass.h"
#include "tinyvk_image.h"

#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <cstdio>

namespace pvdb::gpu {

Cmd DebugDevice::cmd() const
{
    return swapchain.render.cmd(frame);
}

void DebugDevice::init(u32 width, u32 height)
{
    frame = 0;
    swapchain.width = width;
    swapchain.height = height;
    init_device();
    if (width) {
        init_swapchain();
        init_render_pass();
    }
    swapchain.timer = Timer::create(device, vma, 2);
}

void DebugDevice::destroy()
{
    swapchain.timer.destroy(device, vma);
    if (swapchain.width) {
        destroy_render_pass();
        destroy_swapchain();
    }
    destroy_device();
    *this = {};
}

u32 DebugDevice::acquire_image(const Context& ctx)
{
    const u32 image_index = tinyvk::swapchain::from(swapchain.swapchain)
        .acquire_next_image(device, swapchain.acquire.semaphores[frame], VK_NULL_HANDLE, 10000000000);

    if (image_index == UINT32_MAX) {
        destroy_swapchain();
        init_swapchain();
        return image_index;
    }

    swapchain.render.wait(frame);
    swapchain.draw_time = swapchain.timer.delta_time(0, 1);
    auto cmd = swapchain.render.begin(frame);
    swapchain.timer.reset(cmd);
    swapchain.timer.record(cmd, IsGraphics::YES);

    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = render_pass.render_pass;
    begin.framebuffer = render_pass.framebuffers[image_index];
    begin.renderArea = {0, 0, u32(swapchain.width), u32(swapchain.height)};
    begin.clearValueCount = 2;
    begin.pClearValues = (const VkClearValue*)swapchain.clear;
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{0.0f, 0.0f, f32(swapchain.width), f32(swapchain.height), 0.0f, 1.0f};
    vkCmdSetViewport(cmd.vk, 0, 1, &viewport);
    vkCmdSetScissor(cmd.vk, 0, 1, &begin.renderArea);

    ctx.bind(cmd, IsGraphics::YES, frame);

    return image_index;
}

void DebugDevice::present(const Context& ctx, u32 image)
{
    swapchain.timer.record(swapchain.render.cmd(frame), IsGraphics::YES);
    vkCmdEndRenderPass(swapchain.render.cmd(frame));
    swapchain.timer.copy_results(swapchain.render.cmd(frame));
    swapchain.render.submit(ctx, frame);

    bool recreate = tinyvk::swapchain::from(swapchain.swapchain)
        .present_next_image(image, queues.present, {&swapchain.render.semaphores[frame], 1});

    if (recreate) { destroy_swapchain(); init_swapchain(); }

    frame = (frame + 1u) % FRAMES_IN_FLIGHT;
}

void DebugDevice::wait_idle() const
{
    tinyvk::vk_validate(vkDeviceWaitIdle(device), "Failed to wait for device to be idle");
}

void DebugDevice::init_device()
{
    if (swapchain.width) {
        SDL_Init(SDL_INIT_VIDEO);
        swapchain.window = SDL_CreateWindow("DebugDraw",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            swapchain.width, swapchain.height,
            SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN);
    }

    tinyvk::small_vector<const char*> device_ext{}, instance_ext{}, validation{};
    validation.push_back("VK_LAYER_KHRONOS_validation");
    instance_ext.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    device_ext.push_back(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
//    instance_ext.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
//    device_ext.push_back(VK_KHR_MAINTENANCE_3_EXTENSION_NAME);
    device_ext.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

    if (swapchain.width) {
        device_ext.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        u32 ext_count{};
        if (!SDL_Vulkan_GetInstanceExtensions(nullptr, &ext_count, nullptr)) {
            printf("Failed to get instance extensions\n");
            exit(1);
        }
        instance_ext.resize(instance_ext.size() + ext_count);
        if (!SDL_Vulkan_GetInstanceExtensions(nullptr, &ext_count, instance_ext.end() - ext_count)) {
            printf("Failed to get instance extensions\n");
            exit(1);
        }
    }

    // Instance
    auto enable = tinyvk::VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF;
    auto disable = tinyvk::validation_feature_disable_t(
//        tinyvk::VALIDATION_FEATURE_DISABLE_THREAD_SAFETY
//        | tinyvk::VALIDATION_FEATURE_DISABLE_API_PARAMETERS
//        | tinyvk::VALIDATION_FEATURE_DISABLE_OBJECT_LIFETIMES
//        | tinyvk::VALIDATION_FEATURE_DISABLE_CORE_CHECKS
    );
    auto ext = tinyvk::extensions {device_ext, instance_ext, validation, enable, disable};
    instance.instance = tinyvk::instance::create(tinyvk::application_info{}, ext);
    instance.debug_messenger = tinyvk::debug_messenger::create(instance.instance, tinyvk::DEBUG_SEVERITY_ALL, tinyvk::DEBUG_TYPE_VALIDATION_PERF
        , [](tinyvk::vk_debug_severity_t, tinyvk::vk_debug_type_t , tinyvk::debug_messenger::callback_data d, void*) -> u32 {
            if (d->pMessage)
                tinystd::error("%s\n", d->pMessage);
            else if (d->pMessageIdName)
                tinystd::error("%s\n", d->pMessageIdName);
            return u32(0);
        }
        );

    if (swapchain.width) {
        if (!SDL_Vulkan_CreateSurface(swapchain.window, instance.instance, &swapchain.surface)) {
            printf("Failed to create surface\n");
            exit(1);
        }
    }

    // Physical device
    auto devices = tinyvk::physical_devices{instance.instance};
    instance.physical = devices.pick_best(ext, true, swapchain.surface);

    // Logical Device and Queues
    tinyvk::small_vector<tinyvk::queue_request> requests{};
    tinyvk::device_features_t features {tinyvk::FEATURE_VERTEX_STORE_AND_ATOMIC | tinyvk::FEATURE_FRAGMENT_STORE_AND_ATOMIC | tinyvk::FEATURE_SHADER_STORAGE_IMAGE_ARRAY_DYNAMIC_INDEXING};
    auto queue_info = tinyvk::queue_create_info{requests, instance.physical, swapchain.surface};
    device = tinyvk::device::create(instance.instance, instance.physical, queue_info, ext, &features, &vma);
    auto& q_compute = queue_info.queues[tinyvk::QUEUE_COMPUTE][0];
    auto& q_graphics = queue_info.queues[tinyvk::QUEUE_GRAPHICS][0];
    queues.family_compute = q_compute.family;
    queues.family_graphics = q_graphics.family;
    vkGetDeviceQueue(device, q_compute.family, q_compute.index, &queues.compute);
    vkGetDeviceQueue(device, q_graphics.family, q_graphics.index, &queues.graphics);
    queues.present = tinyvk::queue_collection{device, requests, queue_info}.present();

    // Swapchain present sync
    if (swapchain.width) {
        swapchain.acquire = Submit::create(device, queues.present, queues.family_graphics, false);
        swapchain.render = Submit::create(device, queues.present, queues.family_graphics);
        swapchain.render.waits_for(swapchain.acquire, gpu::PipelineStage::COLOR_OUTPUT);
    }
}

void DebugDevice::destroy_device()
{
    if (swapchain.surface) {
        swapchain.acquire.destroy();
        swapchain.render.destroy();
        vkDestroySurfaceKHR(instance.instance, swapchain.surface, nullptr);
    }

    tinyvk::device::from(device).destroy(&vma);
    tinyvk::debug_messenger::from(instance.debug_messenger).destroy(instance.instance);
    tinyvk::instance::from(instance.instance).destroy();

    if (swapchain.window) {
        SDL_DestroyWindow(swapchain.window);
        SDL_Quit();
    }

}

void DebugDevice::init_swapchain()
{
    auto support = tinyvk::swapchain_support{instance.physical, swapchain.surface};
    auto supported_extent = support.supported_extent({u32(swapchain.width), u32(swapchain.height)});

    tinyvk::swapchain_desc d{supported_extent};
    d.vsync = false;
    d.image_count = 3;
    d.formats[0] = VkFormat(Image::vk_format(DebugDeviceSwapchain::FORMAT));

    // swapchain and images
    tinyvk::swapchain::images images{};
    swapchain.swapchain = tinyvk::swapchain::create(device, instance.physical, swapchain.surface, images, d);
    for (auto& i: images)
        swapchain.images[swapchain.image_count++] = i;

    // image views
    tinyvk::image_dimensions dim{u32(swapchain.width), u32(swapchain.height), 1, 1};
    dim.is_cubemap = false;
    dim.mip_levels = 1;
    dim.format = d.formats[0];
    for (u32 i = 0; i < swapchain.image_count; ++i)
        swapchain.views[i] = tinyvk::image_view::create(device, {}, swapchain.images[i], dim);
}

void DebugDevice::destroy_swapchain()
{
    for (u32 i = 0; i < swapchain.image_count; ++i)
        tinyvk::image_view::from(swapchain.views[i]).destroy(device);

    tinyvk::swapchain::from(swapchain.swapchain).destroy(device);
}

void DebugDevice::init_render_pass()
{
    auto swapchain_format = VkFormat(Image::vk_format(DebugDeviceSwapchain::FORMAT));

    tinyvk::renderpass_desc::builder desc{};

    auto color = desc.attach({
        {},
        swapchain_format,
        VK_SAMPLE_COUNT_1_BIT,
        VK_ATTACHMENT_LOAD_OP_CLEAR,
        VK_ATTACHMENT_STORE_OP_STORE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    });

    auto depth = desc.attach({
        {},
        VK_FORMAT_D32_SFLOAT,
        VK_SAMPLE_COUNT_1_BIT,
        VK_ATTACHMENT_LOAD_OP_CLEAR,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    });

    auto s = desc.subpass();
    s.output(color);
    s.output(depth);

    render_pass.render_pass = tinyvk::renderpass::create(device, desc.build());

    for (u32 i = 0; i < swapchain.image_count; ++i) {
        auto& b = render_pass.depth_buffers[i];
        tinyvk::image_dimensions dim{};
        b.image = tinyvk::image::create(vma, b.alloc, {
            {u32(swapchain.width), u32(swapchain.height)},
            VK_FORMAT_D32_SFLOAT,
            tinyvk::IMAGE_DEPTH_STENCIL
        }, &dim);
        b.view = tinyvk::image_view::create(device, {}, b.image, dim);
    }

    for (u32 i = 0; i < swapchain.image_count; ++i) {
        VkImageView views[2]{swapchain.views[i], render_pass.depth_buffers[i].view};
        render_pass.framebuffers[i] = tinyvk::framebuffer::create(device, render_pass.render_pass, {{&views[0], 2}, u32(swapchain.width), u32(swapchain.height)});
    }
}

void DebugDevice::destroy_render_pass()
{
    tinyvk::renderpass::from(render_pass.render_pass).destroy(device);
    for (u32 i = 0; i < swapchain.image_count; ++i) {
        auto& b = render_pass.depth_buffers[i];
        tinyvk::image_view::from(b.view).destroy(device);
        tinyvk::image::from(b.image).destroy(vma, b.alloc);
    }
    for (u32 i = 0; i < swapchain.image_count; ++i)
        tinyvk::framebuffer::from(render_pass.framebuffers[i]).destroy(device);
}

bool DebugDevice::should_quit() const
{
    static constexpr u32 N_ITER = 20;
    SDL_Event e{};
    bool quit{};
    for (u32 i = 0; i < N_ITER; ++i) {
        if (!SDL_PollEvent(&e)) continue;
        switch (e.type) {
            case SDL_QUIT: quit = true; break;
            case SDL_KEYDOWN:
            case SDL_KEYUP: {
//                e.key.keysym.mod, e.key.repeat != 0
                quit = (e.key.state == SDL_PRESSED && e.key.keysym.scancode == SDL_SCANCODE_ESCAPE);
                if (!quit)
                    events.on_key(e.key.keysym.scancode, e.key.state == SDL_PRESSED, e.key.repeat != 0);
                break;
            }
            case SDL_MOUSEMOTION: {
                events.on_mouse_move(e.motion.x, e.motion.y, e.motion.xrel, e.motion.yrel);
                break;
            }
        }
    }
    return quit;
}

ContextDesc DebugDevice::context_desc() const
{
    return {
        device, vma,
        queues.compute, queues.graphics, queues.family_graphics
    };
}

void DebugDevice::lock_mouse(bool locked) const
{
    SDL_SetRelativeMouseMode(locked ? SDL_TRUE : SDL_FALSE);
}

void DebugDevice::set_title(const char *fmt, ...) const
{
    va_list va;
    va_start(va, fmt);
    char buffer[16000]{};
    const int n = vsprintf_s(buffer, 16000, fmt, va);
    va_end(va);
    buffer[n] = 0;
    SDL_SetWindowTitle(swapchain.window, buffer);
}

}