//
// Created by Djordje on 5/14/2022.
//

#ifndef VDB_GPU_CONTEXT_H
#define VDB_GPU_CONTEXT_H

#include "Objects.h"

extern "C" {
typedef struct VkSubmitInfo VkSubmitInfo;
}


namespace pvdb::gpu {

struct ContextDesc {
    VkDevice        device{};
    VmaAllocator    vma{};
    VkQueue         queue_compute{};
    VkQueue         queue_graphics{};
    u32             queue_family{};
};


struct ContextQueues {
    VkQueue         compute{};
    VkQueue         graphics{};
    u32             family{};
};


struct Context {
    struct Storage;

    VkDevice            device{};
    VmaAllocator        vma{};
    VkPipelineLayout    pipeline_layout{};
    VkDescriptorSet     desc_sets[FRAMES_IN_FLIGHT]{};
    Pass*               passes[8]{};
    ResourceSync        pass_sync[8]{};
    ContextQueues       queues{};
    u32                 frame: FRAMES_IN_FLIGHT;
    u32                 pass_count: 32 - FRAMES_IN_FLIGHT;
    Storage*            storage{};

    void        init(const ContextDesc& desc);
    void        destroy();

    void        add_pass(Pass& pass, IsGraphics g = IsGraphics::NO);
    void        setup();
    void        update();
    void        wait() const;

    Cmd         begin_setup() const;
    void        end_setup() const;
    void        wait_setup() const;

    Buffer      create_buffer(u64 size, BufferType type = BufferType::STORAGE, BufferUsage usage = BufferUsage::GPU) const;
    void        destroy_buffer(Buffer& buffer) const;

    Image       create_image(u32 width, u32 height, u32 depth, ImageFormat format, ImageUsage usage) const;
    void        destroy_image(Image& image) const;

    RenderPass  create_render_pass(u32 width, u32 height, ImageFormat color = ImageFormat::UNDEFINED, bool depth = false) const;
    void        destroy_render_pass(RenderPass& r) const;

    Submit      create_submit(IsGraphics g = IsGraphics::NO) const;
    Pipeline    create_compute(span<const u32> binary) const;
    Pipeline    create_compute(ShaderSrc src) const;
    Pipeline    create_graphics(span<const u32> binary_vert, span<const u32> binary_frag, const Pipeline::Graphics& desc) const;
    Pipeline    create_graphics(ShaderSrc src_vert, ShaderSrc src_frag, const Pipeline::Graphics& desc) const;

    void        bind(VkCommandBuffer cmd, IsGraphics g = IsGraphics::NO, u32 frame = -1u) const;
    void        push_const(VkCommandBuffer cmd, const void* data, u32 size, u32 offset = 0) const;

    enum BindPerFrame {
        BIND_SINGLE,            // array of values that will be used for all descriptor sets
        BIND_PER_FRAME,         // [array0: [v0_frame0, v0_frame1], array1: [v1_frame0, v1_frame1]]
        BIND_PER_FRAME_ARRAY    // [frame0: [v0_frame0, v1_frame0], frame1: [v0_frame1, v1_frame1]]
    };
    void        bind(u32 binding, BufferType type, span<const BufferView> buffers, u32 array_index = 0, BindPerFrame per_frame = {}) const;
    void        bind(u32 binding, ImageLayout layout, span<const Image> images, u32 array_index = 0, BindPerFrame per_frame = {}) const;

    template<typename T>
    void        push_const(VkCommandBuffer cmd, const T& value, u32 offset = 0) const { push_const(cmd, &value, sizeof(T), offset); }

private:
    friend struct Submit;

    void fill_image_infos(void* infos, u32 count) const;
    void submit_sync(VkQueue queue, const VkSubmitInfo& info, VkFence fence) const;
};


template<typename Int>
static const char* to_string(Int i)
{
    static u32  size{};
    static char storage[4092];

    const char* begin = storage + size;
    if (i < Int(0)) { storage[size++] = '-'; i = -i; }

    bool began = false;
    u64 multiple = 1000000000u;
    if (sizeof(Int) > 4) multiple = 10000000000000000000ull;
    while (multiple) {
        const Int m = i / Int(multiple);
        i -= m * multiple;
        const char c = '0' + char(m);
        began |= c != '0';
        if (began) { storage[size++] = c; }
        multiple /= 10u;
    }
    if (!began) storage[size++] = '0';
    storage[size++] = 0;
    return begin;
}

}

#endif //VDB_GPU_CONTEXT_H
