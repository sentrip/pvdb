//
// Created by Djordje on 5/14/2022.
//

#include "Objects.h"
#define TINYVK_USE_VMA
#include "tinyvk_buffer.h"
#include "tinyvk_image.h"
#include "tinyvk_shader.h"
#include "tinyvk_pipeline.h"
#include "tinyvk_command.h"
#include "tinyvk_renderpass.h"
#include "tinyvk_queue.h"

#include <spirv_cross/spirv_cross.hpp>

namespace pvdb::gpu {

//region Buffer

BufferView BufferView::from(const Buffer& buffer, u64 offset, u64 size)
{
    return {buffer.vk, offset, size};
}

bool BufferView::overlaps(const BufferView& b, u64& o, u64& s) const
{
    if ((vk != b.vk) || ((offset < b.offset || offset >= (b.offset + b.size)) && (b.offset < offset || b.offset >= (offset + size))))
        return false;
    o = b.offset >= offset ? b.offset : offset;
    s = std::min(offset + size, b.offset + b.size) - offset;
    return true;
}

Buffer Buffer::create(VmaAllocator vma, u64 size, BufferType type, BufferUsage usage)
{
    Buffer b{};
    tinyvk::buffer_usage_t t{};
    if (type == BufferType::INDEX)          t = tinyvk::BUFFER_INDEX;
    if (type == BufferType::VERTEX)         t = tinyvk::BUFFER_VERTEX;
    if (type == BufferType::INDIRECT)       t = tinyvk::BUFFER_INDIRECT;
    if (type == BufferType::UNIFORM)        t = tinyvk::BUFFER_UNIFORM;
    if (type == BufferType::STORAGE)        t = tinyvk::buffer_usage_t(tinyvk::BUFFER_STORAGE | tinyvk::BUFFER_TRANSFER_SRC | tinyvk::BUFFER_TRANSFER_DST);
    if (type == BufferType::INDIRECT_STORAGE)
        t = tinyvk::buffer_usage_t(tinyvk::BUFFER_INDIRECT | tinyvk::BUFFER_STORAGE | tinyvk::BUFFER_TRANSFER_SRC | tinyvk::BUFFER_TRANSFER_DST);

    tinyvk::vma_usage_t u{};
    if (usage == BufferUsage::CPU)          u = tinyvk::VMA_USAGE_CPU_ONLY;
    if (usage == BufferUsage::CPU_TO_GPU)   u = tinyvk::VMA_USAGE_CPU_TO_GPU;
    if (usage == BufferUsage::GPU_TO_CPU)   u = tinyvk::VMA_USAGE_GPU_TO_CPU;
    if (usage == BufferUsage::GPU)          u = tinyvk::VMA_USAGE_GPU_ONLY;

    auto ct = usage == BufferUsage::GPU ? tinyvk::vma_create_t{} : tinyvk::VMA_CREATE_MAPPED;

    b.offset = 0;
    b.size = size;
    b.vk = tinyvk::buffer::create(vma, b.alloc, {size, t, u, ct}, (void**)&b.data);
    return b;
}

void Buffer::destroy(VmaAllocator vma)
{
    tinyvk::buffer::from(vk).destroy(vma, alloc);
    *this = {};
}

//endregion

//region Image

Image Image::create(VkDevice device, VmaAllocator vma, u32 width, u32 height, u32 depth, ImageFormat format, ImageUsage usage)
{
    Image i{};
    tinyvk::image_usage_t u{};
    if (usage == ImageUsage::COLOR)     u = tinyvk::IMAGE_COLOR;
    if (usage == ImageUsage::DEPTH)     u = tinyvk::IMAGE_DEPTH_STENCIL;
    if (usage == ImageUsage::STORAGE)   u = tinyvk::IMAGE_STORAGE;
    tinyvk::image_dimensions dim{};
    i.vk = tinyvk::image::create(vma, i.alloc, {{width, height, depth}, VkFormat(vk_format(format)), u}, &dim);
    i.view = tinyvk::image_view::create(device, {}, i.vk, dim);
    i.width = width;
    i.height = height;
    i.format = u64(format);
    i.usage = u64(usage);
    return i;
}

void Image::destroy(VkDevice device, VmaAllocator vma)
{
    tinyvk::image::from(vk).destroy(vma, alloc);
    tinyvk::image_view::from(view).destroy(device);
    *this = {};
}

u32 Image::vk_format(ImageFormat format)
{
    if (format == ImageFormat::B8G8R8A8_UNORM) return VK_FORMAT_B8G8R8A8_UNORM;
    if (format == ImageFormat::R8G8B8A8_UNORM) return VK_FORMAT_R8G8B8A8_UNORM;
    if (format == ImageFormat::R32_UINT) return VK_FORMAT_R32_UINT;
    if (format == ImageFormat::D32_SFLOAT) return VK_FORMAT_D32_SFLOAT;
    return {};
}

u32 Image::vk_layout(ImageLayout layout)
{
    if (layout == ImageLayout::GENERAL)     return VK_IMAGE_LAYOUT_GENERAL;
    if (layout == ImageLayout::DEPTH_READ)  return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    if (layout == ImageLayout::DEPTH_WRITE) return VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    if (layout == ImageLayout::COLOR_READ)  return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    if (layout == ImageLayout::COLOR_WRITE) return VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    return {};
}

//endregion

//region RenderPass

RenderPass RenderPass::create(VkDevice device, VmaAllocator vma, u32 width, u32 height, ImageFormat color, bool depth)
{
    RenderPass r{};
    tassert((color != ImageFormat::UNDEFINED || depth) && "RenderPass must be color, depth or both");
    r.width = width;
    r.height = height;
    r.create_render_pass(device, color, depth);
    r.create_images_framebuffers(device, vma, color, depth);
    return r;
}

void RenderPass::destroy(VkDevice device, VmaAllocator vma)
{
    tinyvk::renderpass::from(vk).destroy(device);
    destroy_images_framebuffers(device, vma);
    *this = {};
}

void RenderPass::resize(VkDevice device, VmaAllocator vma, u32 w, u32 h)
{
    destroy_images_framebuffers(device, vma);
    width = w;
    height = h;
    create_images_framebuffers(device, vma, ImageFormat(color[0].format), depth[0].vk != nullptr);
}

void RenderPass::create_render_pass(VkDevice device, ImageFormat c, bool d)
{
    tinyvk::renderpass_desc::builder b{};

    auto s = b.subpass(tinyvk::PIPELINE_GRAPHICS);

    if (c != ImageFormat::UNDEFINED) {
        auto a = b.attach({
            {},
            VkFormat(Image::vk_format(c)),
            VK_SAMPLE_COUNT_1_BIT,
            VK_ATTACHMENT_LOAD_OP_CLEAR,
            VK_ATTACHMENT_STORE_OP_STORE,
            VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            VK_ATTACHMENT_STORE_OP_DONT_CARE,
            VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
        });
        s.output(a);
    }

    if (d) {
        auto a = b.attach({
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
        s.output(a);
    }

    vk = tinyvk::renderpass::create(device, b.build());
}

void RenderPass::create_images_framebuffers(VkDevice device, VmaAllocator vma, ImageFormat c, bool d)
{
    if (c != ImageFormat::UNDEFINED) {
        for (auto& v: color)
            v = Image::create(device, vma, width, height, 1, c, ImageUsage::COLOR);
    }
    if (d) {
        for (auto& v: depth)
            v = Image::create(device, vma, width, height, 1, ImageFormat::D32_SFLOAT, ImageUsage::DEPTH);
    }

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        VkImageView views[2]{color[i].view, depth[i].view};
        u32 view_offset = u32(c == ImageFormat::UNDEFINED);
        usize view_count = u32(c != ImageFormat::UNDEFINED) + u32(d);
        framebuffers[i] = tinyvk::framebuffer::create(device, vk, {{&views[view_offset], view_count}, u32(width), u32(height)});
    }
}

void RenderPass::destroy_images_framebuffers(VkDevice device, VmaAllocator vma)
{
    for (auto fb: framebuffers) tinyvk::framebuffer::from(fb).destroy(device);
    if (color[0].vk) { for (auto& c: color) c.destroy(device, vma); }
    if (depth[0].vk) { for (auto& c: depth) c.destroy(device, vma); }
}

//endregion

//region Cmd

void Cmd::reset() const
{
    tinyvk::vk_validate(vkResetCommandBuffer(vk, {}), "Failed to reset command buffer");
}

void Cmd::begin() const
{
    VkCommandBufferBeginInfo cmd_begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    cmd_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    tinyvk::vk_validate(vkBeginCommandBuffer(vk, &cmd_begin), "Failed to begin command");
}

void Cmd::end() const
{
    tinyvk::vk_validate(vkEndCommandBuffer(vk), "Failed to end command");
}

void Cmd::bind_compute(VkPipeline pipeline) const
{
    vkCmdBindPipeline(vk, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
}

void Cmd::bind_graphics(VkPipeline pipeline) const
{
    vkCmdBindPipeline(vk, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
}

void Cmd::draw(u32 vertex_count) const
{
    vkCmdDraw(vk, vertex_count, 1, 0, 0);
}

void Cmd::draw_indirect_indexed(const BufferView& dst, const BufferView& count, u32 max_draws) const
{
    vkCmdDrawIndexedIndirectCount(vk, dst.vk, dst.offset, count.vk, count.offset, max_draws, sizeof(Draw));
}

void Cmd::dispatch(u32 x, u32 y, u32 z) const
{
    vkCmdDispatch(vk, x, y, z);
}

void Cmd::dispatch_indirect(VkBuffer count, u64 offset) const
{
    vkCmdDispatchIndirect(vk, count, offset);
}

void Cmd::copy_buffer(VkBuffer dst, u64 dst_offset, VkBuffer src, u64 src_offset, u64 size) const
{
    if (size == 0) return;
    VkBufferCopy copy{src_offset, dst_offset, size};
    vkCmdCopyBuffer(vk, src, dst, 1u, &copy);
}

void Cmd::fill_buffer(VkBuffer dst, u64 offset, u64 size, u32 value) const
{
    if (size == 0) return;
    vkCmdFillBuffer(vk, dst, offset, size, value);
}

void Cmd::update_buffer(VkBuffer dst, u64 offset, const void* data, usize size) const
{
    if (size == 0) return;
    vkCmdUpdateBuffer(vk, dst, offset, size, data);
}

void Cmd::begin_render_pass(const RenderPass& r, u32 frame, span<const ClearValue> clear) const
{
    VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    info.renderPass = r.vk;
    info.framebuffer = r.framebuffers[frame];
    info.renderArea = {0, 0, u32(r.width), u32(r.height)};
    info.clearValueCount = clear.size();
    info.pClearValues = (const VkClearValue*)clear.data();
    vkCmdBeginRenderPass(vk, &info, VK_SUBPASS_CONTENTS_INLINE);
}

void Cmd::end_render_pass() const
{
    vkCmdEndRenderPass(vk);
}

template<typename Barrier>
void make_barrier(Barrier& barrier, Access src_access, Access dst_access, const BufferBarrier* buffer, const ImageBarrier* image)
{
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VkAccessFlags(src_access);
    barrier.dstAccessMask = VkAccessFlags(dst_access);

    if constexpr (!std::is_same_v<Barrier, VkMemoryBarrier>) {
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        if constexpr (std::is_same_v<Barrier, VkBufferMemoryBarrier>) {
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.buffer = buffer->buffer.vk;
            barrier.offset = buffer->buffer.offset;
            barrier.size = buffer->buffer.size;
        }
        else {
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.oldLayout = VkImageLayout(Image::vk_layout(image->src_layout));
            barrier.newLayout = VkImageLayout(Image::vk_layout(image->dst_layout));
            barrier.image = image->image;
            barrier.subresourceRange.aspectMask = tinyvk::image::determine_aspect_mask(VkFormat(Image::vk_format(image->format)), true);
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
        }
    }
}

void Cmd::barrier(PipelineStage src, PipelineStage dst, Dependency dep, span<const MemoryBarrier> memory, span<const BufferBarrier> buffer, span<const ImageBarrier> image) const
{
    VkMemoryBarrier mem[32]{};
    VkBufferMemoryBarrier buf[32]{};
    VkImageMemoryBarrier im[32]{};
    tassert(memory.size() <= (sizeof(mem)/sizeof(mem[0])) && "Too many memory barriers");
    tassert(buffer.size() <= (sizeof(buf)/sizeof(buf[0])) && "Too many buffer barriers");
    tassert(image.size() <= (sizeof(im)/sizeof(im[0])) && "Too many image barriers");

    for (u32 i = 0; i < memory.size(); ++i) make_barrier(mem[i], memory[i].src, memory[i].dst, nullptr, nullptr);
    for (u32 i = 0; i < buffer.size(); ++i) make_barrier(buf[i], buffer[i].src, buffer[i].dst, &buffer[i], nullptr);
    for (u32 i = 0; i < image.size(); ++i)  make_barrier(im[i], image[i].src, image[i].dst, nullptr, &image[i]);

    vkCmdPipelineBarrier(vk,
        VkPipelineStageFlags(src),
        VkPipelineStageFlags(dst),
        VkDependencyFlags(dep),
        memory.size(), mem,
        buffer.size(), buf,
        image.size(), im);
}

//endregion

//region Submit

Submit Submit::create(VkDevice device, VkQueue queue, u32 queue_family, bool cmds_and_fences)
{
    Submit s{};
    s.device = device;
    s.queue = queue;
    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkSemaphoreCreateInfo sem_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        tinyvk::vk_validate(vkCreateSemaphore(device, &sem_info, nullptr, &s.semaphores[i]), "Failed to create fence");
        if (cmds_and_fences) {
            tinyvk::vk_validate(vkCreateFence(device, &fence_info, nullptr, &s.fences[i]), "Failed to create fence");

            auto pool = tinyvk::command_pool::create(device, queue_family, tinyvk::CMD_POOL_RESET_INDIVIDUAL);
            pool.allocate(device, {&s.commands[i].vk, 1});
            s.command_pools[i] = pool;
        }
    }
    return s;
}

void Submit::destroy()
{
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        vkDestroySemaphore(device, semaphores[i], nullptr);
        if (fences[i]) {
            vkDestroyFence(device, fences[i], nullptr);
            auto pool = tinyvk::command_pool::from(command_pools[i]);
            pool.free(device, {&commands[i].vk, 1});
            pool.destroy(device);
        }
    }
    *this = {};
}

void Submit::waits_for(const Submit& submit, PipelineStage stage)
{
    const u32 wait_i = wait_count++;
    wait_stages[wait_i] = VkPipelineStageFlags(stage);
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; ++i) {
        wait_sem[i][wait_i] = submit.semaphores[i];
    }
}

Cmd Submit::begin(u32 frame) const
{
    tinyvk::vk_validate(vkResetFences(device, 1, &fences[frame]), "Failed to reset fence");
    commands[frame].reset();
    commands[frame].begin();
    return commands[frame];
}

void Submit::wait(u32 frame) const
{
    tinyvk::vk_validate(vkWaitForFences(device, 1, &fences[frame], true, tinyvk::DEFAULT_TIMEOUT_NANOS), "Failed to reset fence");
}

//endregion

//region Timer

u64 Timer::delta_time(u32 from, u32 to) const
{
    auto* ts = ((const u64*)buffer.data);
    return ts[to] - ts[from];
}

u64 Timer::max_time(u32 from, u32 to) const
{
    u64 mt = 0;
    for (u32 i = from; i < to - 1; ++i)
        if (auto dt = delta_time(i, to); dt > mt)
            mt = dt;
    return mt;
}

Timer Timer::create(VkDevice device, VmaAllocator vma, u32 max_steps)
{
    Timer t{};
    t.max_queries = max_steps;
    t.buffer = Buffer::create(vma, max_steps * sizeof(u64), BufferType::STORAGE, BufferUsage::GPU_TO_CPU);

    VkQueryPoolCreateInfo create_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create_info.queryCount = max_steps;
    tinyvk::vk_validate(vkCreateQueryPool(device, &create_info, nullptr, &t.pool),
        "Failed to create query pool");
    return t;
}

void Timer::destroy(VkDevice device, VmaAllocator vma)
{
    buffer.destroy(vma);
    vkDestroyQueryPool(device, pool, nullptr);
    *this = {};
}

void Timer::reset(VkCommandBuffer cmd)
{
    query = 0;
    tinystd::memset(buffer.data, 0, buffer.size);
    vkCmdResetQueryPool(cmd, pool, 0, max_queries);
}

void Timer::record(VkCommandBuffer cmd, IsGraphics g)
{
    vkCmdWriteTimestamp(cmd,
        g == IsGraphics::YES ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        pool, query++);
}

void Timer::copy_results(VkCommandBuffer cmd) const
{
    vkCmdCopyQueryPoolResults(cmd, pool, 0, query, buffer, 0, sizeof(u64), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
}

//endregion

//region ResourceSync

/// Hazards - R_AFTER_W, W_AFTER_W, W_AFTER_R

static Access access_from_stage(PipelineStage stage, bool read, bool write)
{
    const auto u = Access::UNDEFINED;
    switch (stage) {
        case PipelineStage::EARLY_DEPTH_TESTS   :
        case PipelineStage::LATE_DEPTH_TESTS    : return (read ? Access::DEPTH_READ : u)    | (write ? Access::DEPTH_WRITE : u);
        case PipelineStage::COLOR_OUTPUT        : return (read ? Access::COLOR_READ : u)    | (write ? Access::COLOR_WRITE : u);
        case PipelineStage::COMPUTE             : return (read ? Access::SHADER_READ : u)   | (write ? Access::SHADER_WRITE : u);
        case PipelineStage::HOST                : return (read ? Access::HOST_READ : u)     | (write ? Access::HOST_WRITE : u);
        case PipelineStage::DRAW_INDIRECT       : return (read ? Access::INDIRECT_READ : u);
        case PipelineStage::VERTEX_INPUT        : return (read ? Access::VERTEX_READ : u);
        default                                 : return (read ? Access::MEMORY_READ : u)   | (write ? Access::MEMORY_WRITE : u);
    }
}

static ImageLayout layout_from_usage(ImageUsage usage, bool write)
{
    switch (usage) {
        case ImageUsage::DEPTH      : return write ? ImageLayout::DEPTH_WRITE : ImageLayout::DEPTH_READ;
        case ImageUsage::COLOR      : return write ? ImageLayout::COLOR_WRITE : ImageLayout::COLOR_READ;
        case ImageUsage::STORAGE    : return write ? ImageLayout::GENERAL     : ImageLayout::COLOR_READ;
    }
    return ImageLayout::GENERAL;
}


void ResourceSync::init(void *data, usize size, u32 buffers, u32 images)
{
    tassert(size >= (buffers * sizeof(BufInfo) + images * sizeof(ImageInfo)) && "Not enough storage to fit requested sync infos");
    buf = (BufInfo*)data;
    im = (ImageInfo*)(buf + buffers);
    max_buffers = buffers;
    max_images = images;
}

void ResourceSync::clear()
{
    buffer_count = 0;
    image_count = 0;
}

void ResourceSync::read(VkCommandBuffer cmd, const BufferView& buffer, PipelineStage stage)
{
    read_write(cmd, buffer, stage, true, false);
}

void ResourceSync::write(VkCommandBuffer cmd, const BufferView& buffer, PipelineStage stage)
{
    read_write(cmd, buffer, stage, false, true);
}

void ResourceSync::read(VkCommandBuffer cmd, const Image& image, PipelineStage stage)
{
    read_write(cmd, image, stage, true, false);
}

void ResourceSync::write(VkCommandBuffer cmd, const Image& image, PipelineStage stage)
{
    read_write(cmd, image, stage, false, true);
}

void ResourceSync::read_write(VkCommandBuffer cmd, const BufferView& buffer, PipelineStage stage, bool read, bool write)
{
    check_init();
    BufInfo current{}, prev{};
    current.buffer = buffer;
    current.read = u32(read);
    current.write = u32(write);
    current.src_stage = u32(stage);
    if (insert(current, prev))
        barrier(cmd, prev, buffer, stage, read, write);
}

void ResourceSync::read_write(VkCommandBuffer cmd, const Image& image, PipelineStage stage, bool read, bool write)
{
    check_init();
    ImageInfo current{}, prev{};
    current.image = image.vk;
    current.format = ImageFormat(image.format);
    current.usage = image.usage;
    current.read = u32(read);
    current.write = u32(write);
    current.src_stage = u32(stage);
    if (insert(current, prev))
        barrier(cmd, prev, image, stage, read, write);
}

bool ResourceSync::insert(const BufInfo& info, BufInfo& prev)
{
    for (u32 i = 0; i < buffer_count; ++i) {
        u64 offset{}, size{};
        if ((bool(info.write) || bool(buf[i].write)) && info.buffer.overlaps(buf[i].buffer, offset, size)) {
            prev = buf[i];
            buf[i] = info;
            buf[i].buffer.offset = offset;
            buf[i].buffer.size = size;
            return true;
        }
    }
    tassert(buffer_count < max_buffers && "Too many buffer syncs");
    buf[buffer_count++] = info;
    return false;
}

bool ResourceSync::insert(const ImageInfo& info, ImageInfo& prev)
{
    for (u32 i = 0; i < image_count; ++i) {
        if ((bool(info.write) || bool(im[i].write)) && info.image == im[i].image) {
            prev = im[i];
            im[i] = info;
            return true;
        }
    }
    tassert(image_count < max_images && "Too many image syncs");
    im[image_count++] = info;
    return false;
}

void ResourceSync::check_init() const
{
    tassert(buf && "Must call ResourceSync::init() before using any methods");
}

void ResourceSync::barrier(VkCommandBuffer cmd, const BufInfo& prev, const BufferView& buffer, PipelineStage stage, bool read, bool write)
{
    const bool write_after_read = (prev.read && !prev.write) && write;
    auto src_stage = PipelineStage(prev.src_stage);
    auto src_access = access_from_stage(src_stage, bool(prev.read), bool(prev.write));
    auto dst_access = access_from_stage(stage, read, write);
    MemoryBarrier mem{src_access, dst_access};
    BufferBarrier barrier{src_access, dst_access, buffer};
    Cmd{cmd}.barrier(src_stage, stage, {}, {&mem, usize(write_after_read)}, {&barrier, usize(!write_after_read)});
}

void ResourceSync::barrier(VkCommandBuffer cmd, const ImageInfo& prev, const Image& image, PipelineStage stage, bool read, bool write)
{
    const bool write_after_read = (prev.read && !prev.write) && write;
    auto src_stage = PipelineStage(prev.src_stage);
    auto src_layout = layout_from_usage(ImageUsage(prev.usage), bool(prev.write));
    auto dst_layout = layout_from_usage(ImageUsage(image.usage), write);
    auto src_access = access_from_stage(src_stage, bool(prev.read), bool(prev.write));
    auto dst_access = access_from_stage(stage, read, write);
    MemoryBarrier mem{src_access, dst_access};
    ImageBarrier barrier{src_access, dst_access, src_layout, dst_layout, image.vk, ImageFormat(image.format)};
    Cmd{cmd}.barrier(src_stage, stage, {}, {&mem, usize(write_after_read)}, {}, {&barrier, usize(!write_after_read)});
}

//endregion

//region Pipeline

static inline void debug_print_preprocessed(span<const char> preprocessed)
{
    u32 lines = 0;
    printf("0000: ");
    for (auto c: preprocessed) {
        if (c == '\n') { printf("\n%04u: ", lines++); continue; }
        printf("%c", c);
    }
}

VertexAttribs VertexAttribs::reflect(span<const u32> binary)
{
    VertexAttribs attribs{};
    static const VkFormat vec_size_to_format[4]{
        VK_FORMAT_R32_SFLOAT,
        VK_FORMAT_R32G32_SFLOAT,
        VK_FORMAT_R32G32B32_SFLOAT,
        VK_FORMAT_R32G32B32A32_SFLOAT,
    };
    spirv_cross::Compiler compiler{binary.data(), binary.size()};
    auto all = compiler.get_shader_resources();
    for (auto& input: all.stage_inputs) {
        auto& type = compiler.get_type(input.type_id);
        const u32 loc = compiler.get_decoration(input.id, spv::DecorationLocation);
        // type.width       TODO: Handle format width <= 32 for vertex input attributes?
        for (u32 i = 0; i < type.columns; ++i)
            attribs.formats[loc + i] = vec_size_to_format[type.vecsize-1];
        attribs.count += type.columns;
    }
    return attribs;
}

Pipeline Pipeline::create_compute(VkDevice device, VkPipelineLayout layout, span<const u32> bin)
{
    auto shader = tinyvk::shader_module::create(device, bin);
    tinyvk::pipeline::storage_t<256> st{};
    u32 data[4]{3,2,1,0};
    auto pipeline = tinyvk::pipeline::create(device, tinyvk::pipeline::compute_desc{st, layout, shader, {st, { {0u, &data, 16u} }}});
    shader.destroy(device);
    return {pipeline};
}

Pipeline Pipeline::create_compute(VkDevice device, VkPipelineLayout layout, ShaderSrc src)
{
    tinystd::stack_vector<char, 1024> preprocessed{};
    tinyvk::preprocess_shader_cpp(src.src, preprocessed, {(const tinyvk::shader_macro*)src.macros.data(), src.macros.size()});
    auto bin = tinyvk::compile_shader_glslangvalidator(tinyvk::SHADER_COMPUTE, preprocessed, {}, tinyvk::SHADER_OPTIMIZATION_NONE);
    if (bin.empty()) { debug_print_preprocessed(preprocessed); return {}; }
    tinyvk::reflect_shader_convert_const_array_to_spec_const(bin);
    return create_compute(device, layout, bin);
}

Pipeline Pipeline::create_graphics(VkDevice device, VkPipelineLayout layout, span<const u32> bin_vert, span<const u32> bin_frag, const Graphics& desc)
{
    const u32 subpass = 0;
    const u32 stage_count = 2;
    const auto topology = tinyvk::TOPOLOGY_TRIANGLE_LIST;
    const auto attribs = VertexAttribs::reflect(bin_vert);

    auto shader_vert = tinyvk::shader_module::create(device, bin_vert);
    auto shader_frag = tinyvk::shader_module::create(device, bin_frag);

    tinyvk::pipeline::desc_storage storage{};
    tinyvk::pipeline::graphics_desc d{storage, layout, desc.render_pass, subpass, stage_count,
        attribs.count ? desc.bindings.count() : 0u, attribs.count, topology, {}};

    d.add_stage(shader_vert, tinyvk::SHADER_VERTEX);
    d.add_stage(shader_frag, tinyvk::SHADER_FRAGMENT);

    d.add_dynamic_state(VK_DYNAMIC_STATE_VIEWPORT);
    d.add_dynamic_state(VK_DYNAMIC_STATE_SCISSOR);

    d.rasterizer(VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);

    tinyvk::pipeline::blend_desc blend{{VK_BLEND_FACTOR_ONE}};
    d.blending(storage, {&blend, 1});

    if (desc.depth)
        d.depth(storage);

    if (attribs.count > 0) {
        u32 size[16]{};
        for (u32 i = 0; i < attribs.count; ++i)
            size[desc.bindings.bind[i]] += tinyvk::vk_format_size_bytes(VkFormat(attribs.formats[i]));
        for (u32 i = 0; i < desc.bindings.count(); ++i)
            d.add_vertex_binding(size[i], i >= desc.first_instance_attribute);

        u32 offset[16]{};
        for (u32 i = 0; i < attribs.count; ++i) {
            const auto b = desc.bindings.bind[i];
            const auto f = VkFormat(attribs.formats[i]);
            d.add_vertex_attribute(b, i, f, offset[b]);
            offset[b] += tinyvk::vk_format_size_bytes(f);
        }
    }

    auto pipeline = tinyvk::pipeline::create(device, d);
    shader_vert.destroy(device);
    shader_frag.destroy(device);
    return {pipeline};
}

Pipeline Pipeline::create_graphics(VkDevice device, VkPipelineLayout layout, ShaderSrc src_vert, ShaderSrc src_frag, const Graphics& desc)
{
    tinystd::stack_vector<char, 1024> preprocessed_vert{}, preprocessed_frag{};
    tinyvk::preprocess_shader_cpp(src_vert.src, preprocessed_vert, {(const tinyvk::shader_macro*)src_vert.macros.data(), src_vert.macros.size()});
    tinyvk::preprocess_shader_cpp(src_frag.src, preprocessed_frag, {(const tinyvk::shader_macro*)src_frag.macros.data(), src_frag.macros.size()});
    auto bin_vert = tinyvk::compile_shader_glslangvalidator(tinyvk::SHADER_VERTEX, preprocessed_vert, {}, tinyvk::SHADER_OPTIMIZATION_NONE);
    if (bin_vert.empty()) { debug_print_preprocessed(preprocessed_vert); return {}; }
    tinyvk::reflect_shader_convert_const_array_to_spec_const(bin_vert);

    auto bin_frag = tinyvk::compile_shader_glslangvalidator(tinyvk::SHADER_FRAGMENT, preprocessed_frag, {}, tinyvk::SHADER_OPTIMIZATION_NONE);
    if (bin_frag.empty()) { debug_print_preprocessed(preprocessed_frag); return {}; }
    tinyvk::reflect_shader_convert_const_array_to_spec_const(bin_frag);

    return create_graphics(device, layout, bin_vert, bin_frag, desc);
}

void Pipeline::destroy(VkDevice device)
{
    tinyvk::pipeline::from(vk).destroy(device);
    *this = {};
}

//endregion

}
