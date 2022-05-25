//
// Created by Djordje on 5/14/2022.
//

#include "Context.h"
#include "Pass.h"

#include <new>
#include "tinyvk_command.h"
#include "tinyvk_descriptor.h"
#include "tinyvk_pipeline.h"
#include "tinyvk_shader.h"


namespace pvdb::gpu {

struct Context::Storage {
    static constexpr usize PASS_SYNC_SIZE           = 4096;
    static constexpr usize BUFFER_SYNC_COUNT        = 32;
    static constexpr usize IMAGE_SYNC_COUNT         = 16;
    static constexpr usize NULL_RESOURCE_COUNT      = 32;
    tinyvk::command_pool                            setup_pool{};
    Cmd                                             setup_cmd{};
    VkFence                                         setup_fence{};
    Buffer                                          null_buffer{};
    Image                                           null_image{};
    tinyvk::descriptor_pool_allocator               desc_pool_alloc{};
    tinyvk::descriptor_set_layout                   descriptor_set_layout{};
    tinystd::stack_vector<tinyvk::descriptor, 64>   descriptors{};
    tinystd::stack_vector<VkWriteDescriptorSet, 64> descriptor_writes[FRAMES_IN_FLIGHT]{};
    tinystd::stack_vector<Pipeline, 32>             pipelines{};
    VkDescriptorBufferInfo                          null_buffers[NULL_RESOURCE_COUNT]{};
    VkDescriptorImageInfo                           null_images[NULL_RESOURCE_COUNT]{};
    u8                                              pass_sync_storage[PASS_SYNC_SIZE]{};
    usize                                           pass_sync_storage_used{};

    template<typename VkT, typename T>
    void bind(u32 binding, tinyvk::descriptor_type_t type, span<const T> values, u32 array_index, BindPerFrame per_frame);
};

static constexpr tinyvk::descriptor_pool_size POOL_SIZE = [](){
    tinyvk::descriptor_pool_size s{256};
    for (auto& v: s.sizes) v = 0.0f;
    s.sizes[tinyvk::DESCRIPTOR_STORAGE_BUFFER] = 1.0f;
    s.sizes[tinyvk::DESCRIPTOR_UNIFORM_BUFFER] = 1.0f;
    s.sizes[tinyvk::DESCRIPTOR_STORAGE_IMAGE]  = 1.0f;
    return s;
}();

void Context::init(const ContextDesc& desc)
{
    device = desc.device;
    vma = desc.vma;
    queues.compute = desc.queue_compute;
    queues.graphics = desc.queue_graphics;
    queues.family = desc.queue_family;
    frame = 0;
    pass_count = 0;

    storage = new Storage;
    storage->desc_pool_alloc.init(device, POOL_SIZE, 8);

    storage->setup_pool = tinyvk::command_pool::create(device, desc.queue_family, {});
    storage->setup_pool.allocate(device, {&storage->setup_cmd.vk, 1});
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
    tinyvk::vk_validate(vkCreateFence(device, &fence_info, nullptr, &storage->setup_fence), "Failed to create fence");

    storage->null_buffer = Buffer::create(vma, 512, BufferType::STORAGE, BufferUsage::GPU);
    storage->null_image = Image::create(device, vma, 16, 16, 16, ImageFormat::R32_UINT, ImageUsage::STORAGE);
    for (auto& b: storage->null_buffers) b = {storage->null_buffer.vk, 0, ~0ull};
    for (auto& i: storage->null_images)  i = {{}, storage->null_image.view, VK_IMAGE_LAYOUT_GENERAL};

    // This shouldn't be necessary
    auto cmd = begin_setup();
    ImageBarrier im{Access::UNDEFINED, Access::SHADER_WRITE, ImageLayout{}, ImageLayout::GENERAL, storage->null_image.vk, ImageFormat::R32_UINT};
    cmd.barrier(PipelineStage::COMPUTE, PipelineStage::COMPUTE, {}, {}, {}, {&im, 1});
    end_setup();
    wait_setup();
}

void Context::destroy()
{
    storage->null_image.destroy(device, vma);
    storage->null_buffer.destroy(vma);

    vkDestroyFence(device, storage->setup_fence, nullptr);
    storage->setup_pool.free(device, {&storage->setup_cmd.vk, 1});
    storage->setup_pool.destroy(device);

    for (u32 i = 0; i < pass_count; ++i)
        passes[i]->destroy();

    for (auto& p: storage->pipelines)
        p.destroy(device);

    storage->descriptor_set_layout.destroy(device);
    tinyvk::pipeline_layout::from(pipeline_layout).destroy(device);

    storage->desc_pool_alloc.destroy(device);
    delete storage;
    *this = {};
}

Cmd Context::begin_setup() const
{
    tinyvk::vk_validate(vkWaitForFences(device, 1, &storage->setup_fence, true, tinyvk::DEFAULT_TIMEOUT_NANOS), "Failed to wait for fences");
    tinyvk::vk_validate(vkResetFences(device, 1, &storage->setup_fence), "Failed to wait for fences");
    tinyvk::vk_validate(vkResetCommandPool(device, storage->setup_pool, {}), "Failed to reset command pool");
    storage->setup_cmd.begin();
    return storage->setup_cmd;
}

void Context::end_setup() const
{
    storage->setup_cmd.end();
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.commandBufferCount = 1;
    info.pCommandBuffers = &storage->setup_cmd.vk;
    submit_sync(queues.graphics, info, storage->setup_fence);
}

void Context::wait_setup() const
{
    tinyvk::vk_validate(vkWaitForFences(device, 1, &storage->setup_fence, true, tinyvk::DEFAULT_TIMEOUT_NANOS), "Failed to wait for fences");
}

void Context::add_pass(Pass& pass, IsGraphics g)
{
    tassert(pass_count < (sizeof(passes)/sizeof(passes[0])) && "Too many passes");
    auto* begin = storage->pass_sync_storage + storage->pass_sync_storage_used;
    auto size = ResourceSync::allocation_size(Storage::BUFFER_SYNC_COUNT, Storage::IMAGE_SYNC_COUNT);
    pass_sync[pass_count].init(begin, size, Storage::BUFFER_SYNC_COUNT, Storage::IMAGE_SYNC_COUNT);
    storage->pass_sync_storage_used += size;
    pass.init(*this, pass_sync[pass_count], g);
    passes[pass_count++] = &pass;
}

void Context::setup()
{
    if (pipeline_layout) return;

    VkDescriptorImageInfo image_infos[1024]{};
    fill_image_infos(image_infos, 1024);

    storage->descriptor_set_layout = tinyvk::descriptor_set_layout::create(device, storage->descriptors);
    storage->desc_pool_alloc.allocate(device, desc_sets, storage->descriptor_set_layout);
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        tinyvk::descriptor_set::write(device, storage->descriptor_writes[i], desc_sets[i]);
        storage->descriptor_writes[i].clear();
    }

    auto pc = tinyvk::push_constant_range{0, 128, tinyvk::SHADER_ALL};
    pipeline_layout = tinyvk::pipeline_layout::create(device, {&storage->descriptor_set_layout.vk, 1}, {&pc, 1});

    for (u32 i = 0; i < pass_count; ++i)
        passes[i]->setup();
}

void Context::update()
{
    if (!pipeline_layout) {
        setup();
    } else {
        VkDescriptorImageInfo image_infos[1024]{};
        if (!storage->descriptor_writes[frame].empty()) {
            fill_image_infos(image_infos, 1024);
            tinyvk::descriptor_set::write(device, storage->descriptor_writes[frame], desc_sets[frame]);
            storage->descriptor_writes[frame].clear();
        }
    }

    for (u32 i = 0; i < pass_count; ++i)
        passes[i]->execute();

    frame = (frame + 1u) % FRAMES_IN_FLIGHT;
}

void Context::wait() const
{
    for (u32 i = 0; i < pass_count; ++i)
        passes[i]->wait();
}

void Context::fill_image_infos(void* infos, u32 count) const
{
    auto* image_infos = static_cast<VkDescriptorImageInfo*>(infos);
    u32 image_count = 0;
    const u32 f_begin = pipeline_layout ? frame : 0u;
    const u32 f_end   = pipeline_layout ? (frame + 1u) : FRAMES_IN_FLIGHT;
    for (u32 f = f_begin; f < f_end; ++f) {
        for (auto& w: storage->descriptor_writes[f]) {
            if (w.pNext == nullptr) continue;
            const u32 info = u32(((u8*)w.pNext) - ((u8*)nullptr));
            auto layout = ImageLayout(info & 0xffffu);
            auto vk_layout = VkImageLayout(Image::vk_layout(layout));
            w.pNext = nullptr;
            if ((info >> 16u) != 0u) {
                for (u32 i = 0; i < w.descriptorCount; ++i)
                    ((VkDescriptorImageInfo*)w.pImageInfo)->imageLayout = vk_layout;
            } else {
                auto* images = (const Image*)w.pImageInfo;
                auto* begin = image_infos + image_count;
                for (u32 i = 0; i < w.descriptorCount; ++i) {
                    begin[i].imageLayout = vk_layout;
                    begin[i].imageView = images[i].view;
                    begin[i].sampler = nullptr;
                }
                w.pImageInfo = begin;
                image_count += w.descriptorCount;
                tassert(image_count <= count && "Too many images bound");
            }
        }
    }
}

template<typename VkT, typename T>
void Context::Storage::bind(u32 binding, tinyvk::descriptor_type_t type, span<const T> values, u32 array_index, BindPerFrame per_frame)
{
    static constexpr auto do_write = [](auto& vec, u32 binding, tinyvk::descriptor_type_t type, span<const VkT> views, u32 array_index){
        if constexpr(tinystd::is_same_v<VkT, VkDescriptorBufferInfo>)
            tinyvk::descriptor_set::write_buffers(vec, binding, type, views, array_index);
        else
            tinyvk::descriptor_set::write_images(vec, binding, type, views, array_index);
    };

    const VkT* null{};
    if constexpr(tinystd::is_same_v<VkT, VkDescriptorBufferInfo>)
        null = null_buffers;
    else
        null = null_images;

    const u32 step_count = values.data() != nullptr ? 1u : (1u + (values.size() - 1u) / Storage::NULL_RESOURCE_COUNT);
    for (u32 step = 0; step < step_count; ++step) {
        auto views = values.data() != nullptr
            ? span<const VkT>{(const VkT*)values.data(), values.size()}
            : span<const VkT>{null, Storage::NULL_RESOURCE_COUNT};

        if (!descriptor_set_layout.vk)
            descriptors.push_back({binding, type, u32(views.size()), tinyvk::SHADER_ALL});

        if (per_frame == BIND_SINGLE) {
            for (auto& w: descriptor_writes)
                do_write(w, binding, type, views, array_index);
        }
        else if (per_frame == BIND_PER_FRAME_ARRAY) {
            const usize per_frame_count = values.size() / FRAMES_IN_FLIGHT;
            for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
                decltype(views) vw{views.data() + (i * per_frame_count), per_frame_count};
                do_write(descriptor_writes[i], binding, type, vw, array_index);
            }
        }
        else {
            const usize n = values.size() / FRAMES_IN_FLIGHT;
            for (auto& w: descriptor_writes) {
                for (u32 j = 0; j < n; ++j) {
                    decltype(views) vw{views.data() + (j * FRAMES_IN_FLIGHT), FRAMES_IN_FLIGHT};
                    do_write(w, binding, type, vw, array_index);
                }
            }
        }
    }
}

void Context::bind(u32 binding, BufferType type, span<const BufferView> buffers, u32 array_index, BindPerFrame per_frame) const
{
    tinyvk::descriptor_type_t t{};
    if      (type == BufferType::UNIFORM) t = tinyvk::DESCRIPTOR_UNIFORM_BUFFER;
    else                                  t = tinyvk::DESCRIPTOR_STORAGE_BUFFER;
    storage->bind<VkDescriptorBufferInfo>(binding, t, buffers, array_index, per_frame);
}

void Context::bind(u32 binding, ImageLayout layout, span<const Image> images, u32 array_index, BindPerFrame per_frame) const
{
    const u32 count = storage->descriptor_writes[0].size();
    storage->bind<VkDescriptorImageInfo>(binding, tinyvk::DESCRIPTOR_STORAGE_IMAGE, images, array_index, per_frame);
    const u32 new_count = storage->descriptor_writes[0].size();
    for (u32 i = count; i < new_count; ++i) {
        for (auto& w: storage->descriptor_writes)
            w[i].pNext = ((u8*)nullptr) + (u32(u32(images.data() == nullptr) << 16u) | u32(layout));
    }
}

void Context::bind(VkCommandBuffer cmd, IsGraphics g, u32 f) const
{
    vkCmdBindDescriptorSets(cmd,
        g == IsGraphics::YES ? VK_PIPELINE_BIND_POINT_GRAPHICS : VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout,
        0,
        1, &desc_sets[f == -1u ? frame : f],
        0, nullptr);
}

void Context::push_const(VkCommandBuffer cmd, const void *data, u32 size, u32 offset) const
{
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_ALL, offset, size, data);
}


Buffer Context::create_buffer(u64 size, BufferType type, BufferUsage usage) const
{
    return Buffer::create(vma, size, type, usage);
}

void Context::destroy_buffer(Buffer& buffer) const
{
    buffer.destroy(vma);
}

Image Context::create_image(u32 width, u32 height, u32 depth, ImageFormat format, ImageUsage usage) const
{
    return Image::create(device, vma, width, height, depth, format, usage);
}

void Context::destroy_image(Image& image) const
{
    image.destroy(device, vma);
}

RenderPass Context::create_render_pass(u32 width, u32 height, ImageFormat color, bool depth) const
{
    return RenderPass::create(device, vma, width, height, color, depth);
}

void Context::destroy_render_pass(RenderPass& r) const
{
    r.destroy(device, vma);
}


Submit Context::create_submit(IsGraphics g) const
{
    bool gfx = g == IsGraphics::YES;
    return Submit::create(device, gfx ? queues.graphics : queues.compute, queues.family, true);
}

Pipeline Context::create_compute(span<const u32> binary) const
{
    tassert(pipeline_layout && "Must call Context::setup() once before creating pipelines");
    auto p = Pipeline::create_compute(device, pipeline_layout, binary);
    storage->pipelines.push_back(p);
    return p;
}

Pipeline Context::create_compute(ShaderSrc src) const
{
    tassert(pipeline_layout && "Must call Context::setup() once before creating pipelines");
    auto p = Pipeline::create_compute(device, pipeline_layout, src);
    storage->pipelines.push_back(p);
    return p;
}

Pipeline Context::create_graphics(span<const u32> binary_vert, span<const u32> binary_frag, const Pipeline::Graphics& desc) const
{
    tassert(pipeline_layout && "Must call Context::setup() once before creating pipelines");
    auto p = Pipeline::create_graphics(device, pipeline_layout, binary_vert, binary_frag, desc);
    storage->pipelines.push_back(p);
    return p;
}

Pipeline Context::create_graphics(ShaderSrc src_vert, ShaderSrc src_frag, const Pipeline::Graphics& desc) const
{
    tassert(pipeline_layout && "Must call Context::setup() once before creating pipelines");
    auto p = Pipeline::create_graphics(device, pipeline_layout, src_vert, src_frag, desc);
    storage->pipelines.push_back(p);
    return p;
}

void Context::submit_sync(VkQueue queue, const VkSubmitInfo& info, VkFence fence) const
{
    if (device == nullptr) return;
    // TODO: External queue sync
    tinyvk::vk_validate(vkQueueSubmit(queue, 1, &info, fence), "Failed to submit to queue");
}

/// NOTE: To avoid including "Context.h" in "Objects.cpp" so it can be used in a single line of code, this method is defined here
/// where Context is defined and since all of the other objects used are already included, there are no extra includes generated
void Submit::submit(const Context& ctx, u32 frame, bool is_last) const
{
    commands[frame].end();
    VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    info.signalSemaphoreCount = u32(!is_last);
    info.waitSemaphoreCount = wait_count;
    info.pSignalSemaphores = &semaphores[frame];
    info.pWaitSemaphores = wait_sem[frame];
    info.pWaitDstStageMask = wait_stages;
    info.pCommandBuffers = (const VkCommandBuffer*)&commands[frame];
    info.commandBufferCount = 1;
    ctx.submit_sync(queue, info, fences[frame]);
}

}
