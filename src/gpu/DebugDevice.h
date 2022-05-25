//
// Created by Djordje on 5/14/2022.
//

#ifndef VDB_GPU_DEBUGDEVICE_H
#define VDB_GPU_DEBUGDEVICE_H

extern "C" {
typedef struct SDL_Window SDL_Window;
}

#include "Objects.h"
#include <functional>

namespace pvdb {
struct FPSCamera;
}

namespace pvdb::gpu {

enum {
    SWAPCHAIN_IMAGES = 8
};


struct DebugDeviceInstance {
    VkInstance                  instance{};
    VkPhysicalDevice            physical{};
    VkDebugUtilsMessengerEXT    debug_messenger{};
};


struct DebugDeviceQueues {
    VkQueue     compute{};
    VkQueue     graphics{};
    VkQueue     present{};
    u32         family_compute{};
    u32         family_graphics{};
};


struct DebugDeviceSwapchain {
    static constexpr ImageFormat FORMAT = ImageFormat::B8G8R8A8_UNORM;

    SDL_Window*     window{};
    VkSwapchainKHR  swapchain{};
    VkSurfaceKHR    surface{};
    u64             image_count: 8;
    u64             width: 28;
    u64             height: 28;
    VkImage         images[SWAPCHAIN_IMAGES]{};
    VkImageView     views[SWAPCHAIN_IMAGES]{};
    ClearValue      clear[2]{{0.2, 0.2, 0.2, 0.2}, {1.0, 0}}; // color, depth
    Submit          acquire{};
    Submit          render{};
    Timer           timer{};
    u64             draw_time{};
};


struct DebugDeviceRenderPass {
    VkRenderPass    render_pass{};
    VkFramebuffer   framebuffers[SWAPCHAIN_IMAGES]{};
    struct {
        VkImage         image{};
        VkImageView     view{};
        VmaAllocation   alloc{};
    }               depth_buffers[SWAPCHAIN_IMAGES]{};
};


struct DebugDeviceEvents {
    std::function<void(i32 px, i32 py, i32 mx, i32 my)> on_mouse_move{[](i32,i32,i32,i32){}};
    std::function<void(i32 k, bool p, bool r)> on_key{[](i32, bool, bool){}};
};


struct DebugDevice {
    VkDevice                device{};
    VmaAllocator            vma{};
    u32                     frame{};
    DebugDeviceQueues       queues{};
    DebugDeviceInstance     instance{};
    DebugDeviceSwapchain    swapchain{};
    DebugDeviceRenderPass   render_pass{};
    DebugDeviceEvents       events{};

    void init(u32 width = 0, u32 height = 0);
    void destroy();

    bool should_quit() const;
    Cmd  cmd() const;
    u32  acquire_image(const Context& ctx);
    void present(const Context& ctx, u32 image);
    void wait_idle() const;

    ContextDesc context_desc() const;

    void lock_mouse(bool locked) const;
    void set_title(const char* fmt, ...) const;

private:
    void init_device();
    void destroy_device();
    void init_swapchain();
    void destroy_swapchain();
    void init_render_pass();
    void destroy_render_pass();
};

}

namespace pvdb {

enum Key {
    KEY_A = 4,
    KEY_B = 5,
    KEY_C = 6,
    KEY_D = 7,
    KEY_E = 8,
    KEY_F = 9,
    KEY_G = 10,
    KEY_H = 11,
    KEY_I = 12,
    KEY_J = 13,
    KEY_K = 14,
    KEY_L = 15,
    KEY_M = 16,
    KEY_N = 17,
    KEY_O = 18,
    KEY_P = 19,
    KEY_Q = 20,
    KEY_R = 21,
    KEY_S = 22,
    KEY_T = 23,
    KEY_U = 24,
    KEY_V = 25,
    KEY_W = 26,
    KEY_X = 27,
    KEY_Y = 28,
    KEY_Z = 29,
    KEY_RETURN = 40,
    KEY_ESCAPE = 41,
    KEY_BACKSPACE = 42,
    KEY_TAB = 43,
    KEY_SPACE = 44,
    KEY_LCTRL = 224,
    KEY_LSHIFT = 225,
    KEY_LALT = 226,
    KEY_LGUI = 227,
    KEY_RCTRL = 228,
    KEY_RSHIFT = 229,
    KEY_RALT = 230,
    KEY_RGUI = 231,
};

}

#endif //VDB_GPU_DEBUGDEVICE_H
