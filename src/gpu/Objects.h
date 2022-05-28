//
// Created by Djordje on 5/14/2022.
//

#ifndef VDB_GPU_OBJECTS_H
#define VDB_GPU_OBJECTS_H

#include "fwd.h"

namespace pvdb::gpu {

enum class BufferType {
    STORAGE,
    INDEX,
    VERTEX,
    INDIRECT,
    UNIFORM,
    INDIRECT_STORAGE
};

enum class BufferUsage {
    CPU,
    CPU_TO_GPU,
    GPU_TO_CPU,
    GPU,
};

enum class ImageFormat {
    UNDEFINED,
    B8G8R8A8_UNORM,     // swapchain
    R8G8B8A8_UNORM,     // color
    R32_UINT,           // storage unsigned integer
    R32G32_UINT,        // storage unsigned integer
    R32G32B32_UINT,     // storage unsigned integer
    R32G32B32A32_UINT,  // storage unsigned integer
    D32_SFLOAT,         // depth
};

enum class ImageLayout {
    UNDEFINED,
    COLOR_READ,
    COLOR_WRITE,
    DEPTH_READ,
    DEPTH_WRITE,
    GENERAL,
};

enum class ImageUsage {
    DEPTH,
    COLOR,
    STORAGE,
};

enum class ShaderType {
    COMPUTE,
    VERTEX,
    FRAGMENT
};


enum class Access: u32 {
    UNDEFINED = 0,
    INDIRECT_READ = 0x00000001,
    INDEX_READ = 0x00000002,
    VERTEX_READ = 0x00000004,
    UNIFORM_READ = 0x00000008,
    INPUT_READ = 0x00000010,
    SHADER_READ = 0x00000020,
    SHADER_WRITE = 0x00000040,
    COLOR_READ = 0x00000080,
    COLOR_WRITE = 0x00000100,
    DEPTH_READ = 0x00000200,
    DEPTH_WRITE = 0x00000400,
    TRANSFER_READ = 0x00000800,
    TRANSFER_WRITE = 0x00001000,
    HOST_READ = 0x00002000,
    HOST_WRITE = 0x00004000,
    MEMORY_READ = 0x00008000,
    MEMORY_WRITE = 0x00010000,
};
constexpr Access operator|(Access l, Access r) { return Access(u32(l) | u32(r)); }
constexpr Access operator&(Access l, Access r) { return Access(u32(l) & u32(r)); }


enum class PipelineStage: u32 {
    TOP               = 0x00000001,
    DRAW_INDIRECT     = 0x00000002,
    VERTEX_INPUT      = 0x00000004,
    VERTEX            = 0x00000008,
    FRAGMENT          = 0x00000080,
    EARLY_DEPTH_TESTS = 0x00000100,
    LATE_DEPTH_TESTS  = 0x00000200,
    COLOR_OUTPUT      = 0x00000400,
    COMPUTE           = 0x00000800,
    TRANSFER          = 0x00001000,
    BOTTOM            = 0x00002000,
    HOST              = 0x00004000,
    ALL_GRAPHICS      = 0x00008000,
    ALL_COMMANDS      = 0x00010000,
};
constexpr PipelineStage operator|(PipelineStage l, PipelineStage r) { return PipelineStage(u32(l) | u32(r)); }
constexpr PipelineStage operator&(PipelineStage l, PipelineStage r) { return PipelineStage(u32(l) & u32(r)); }


enum class Dependency: u32 {
    BY_REGION       = 0x00000001,
    VIEW_LOCAL      = 0x00000002,
    DEVICE_GROUP    = 0x00000004,
};
constexpr Dependency operator|(Dependency l, Dependency r) { return Dependency(u32(l) | u32(r)); }
constexpr Dependency operator&(Dependency l, Dependency r) { return Dependency(u32(l) & u32(r)); }

}

namespace pvdb::gpu {

//region Buffer/Image/Renderpass

struct BufferView {
    VkBuffer            vk{};
    u64                 offset{};
    u64                 size{};

    static BufferView   from(const Buffer& buffer, u64 offset = 0, u64 size = ~0ull);
    bool overlaps(const BufferView& buffer, u64& offset, u64& size) const;
};


struct Buffer : BufferView {
    u8*                 data{};
    VmaAllocation       alloc{};

    operator VkBuffer() const { return vk; }

    static Buffer       create(VmaAllocator vma, u64 size, BufferType type, BufferUsage usage);
    void                destroy(VmaAllocator vma);
};


struct Image {
    VkImage             vk{};
    VkImageView         view{};
    VmaAllocation       alloc{};
    u64                 width: 16;
    u64                 height: 16;
    u64                 format: 16;
    u64                 usage: 16;

    operator VkImage() const { return vk; }
    operator VkImageView() const { return view; }

    static Image        create(VkDevice device, VmaAllocator vma, u32 width, u32 height, u32 depth, ImageFormat format, ImageUsage usage);
    void                destroy(VkDevice device, VmaAllocator vma);

    static u32          vk_format(ImageFormat format);
    static u32          vk_layout(ImageLayout layout);
};


struct RenderPass {
    VkRenderPass        vk{};
    u64                 width: 30;
    u64                 height: 30;
    VkFramebuffer       framebuffers[FRAMES_IN_FLIGHT]{};
    Image               color[FRAMES_IN_FLIGHT]{};
    Image               depth[FRAMES_IN_FLIGHT]{};

    operator VkRenderPass() const { return vk; }
    VkFramebuffer operator[](usize i) const { return framebuffers[i]; }

    static RenderPass   create(VkDevice device, VmaAllocator vma, u32 width, u32 height, ImageFormat color = ImageFormat::UNDEFINED, bool depth = false);
    void                destroy(VkDevice device, VmaAllocator vma);
    void                resize(VkDevice device, VmaAllocator vma, u32 width, u32 height);

private:
    void create_render_pass(VkDevice device, ImageFormat color, bool depth);
    void create_images_framebuffers(VkDevice device, VmaAllocator vma, ImageFormat color, bool depth);
    void destroy_images_framebuffers(VkDevice device, VmaAllocator vma);
};

//endregion

//region Cmd

union ClearValue {
    struct {
        f32 r, g, b, a;
    };
    struct {
        f32 depth;
        u32 stencil;
    };
};


struct MemoryBarrier {
    Access      src{};
    Access      dst{};
};


struct BufferBarrier : MemoryBarrier {
    BufferView  buffer{};
};


struct ImageBarrier : MemoryBarrier {
    ImageLayout src_layout{};
    ImageLayout dst_layout{};
    VkImage     image{};
    ImageFormat format{};
};


struct Cmd {
    VkCommandBuffer     vk{};

    operator VkCommandBuffer() const { return vk; }

    void                reset() const;
    void                begin() const;
    void                end() const;
    void                bind_compute(VkPipeline pipeline) const;
    void                bind_graphics(VkPipeline pipeline) const;
    void                begin_render_pass(const RenderPass& r, u32 frame, span<const ClearValue> clear = {}) const;
    void                end_render_pass() const;

    void                draw(u32 vertex_count) const;
    void                draw_indirect_indexed(const BufferView& dst, const BufferView& count, u32 max_draws) const;
    void                dispatch(u32 x = 1, u32 y = 1, u32 z = 1) const;
    void                dispatch_indirect(VkBuffer count, u64 offset = 0) const;
    void                copy_buffer(VkBuffer dst, u64 dst_offset, VkBuffer src, u64 src_offset, u64 size) const;
    void                fill_buffer(VkBuffer dst, u64 offset, u64 size, u32 value) const;
    void                update_buffer(VkBuffer dst, u64 offset, const void* data, usize size) const;
    template<typename T = u8>
    void                update_buffer(VkBuffer dst, u64 offset, span<const T> data) const { update_buffer(dst, offset, (const void*)data.data(), data.size() * sizeof(T)); }

    void                barrier(PipelineStage src, PipelineStage dst, Dependency dep = {},
                                span<const MemoryBarrier> memory = {},
                                span<const BufferBarrier> buffer = {},
                                span<const ImageBarrier> image = {}) const;

    struct Draw {
        u32                        vertex_count{};
        u32                        instance_count{1};
        u32                        first_vertex{};
        u32                        first_instance{};
    };
};


struct Submit {
    static constexpr u32 MAX = 4;
    VkDevice                device{};
    VkQueue                 queue{};
    PerFrame<VkFence>       fences{};
    PerFrame<VkSemaphore>   semaphores{};
    VkSemaphore             wait_sem[FRAMES_IN_FLIGHT][MAX]{};
    PerFrame<Cmd>           commands{};
    PerFrame<VkCommandPool> command_pools{};
    u32                     wait_stages[MAX]{};
    u32                     wait_count{};

    static Submit create(VkDevice device, VkQueue queue, u32 queue_family, bool cmds_and_fences = true);
    void destroy();

    void waits_for(const Submit& submit, PipelineStage stage);

    Cmd  cmd(u32 frame) const { return commands[frame]; }
    Cmd  begin(u32 frame) const;
    void submit(const Context& ctx, u32 frame, bool is_last = false) const;
    void wait(u32 frame) const;
};


struct Timer {
    Timer() = default;

    static Timer        create(VkDevice device, VmaAllocator vma, u32 max_steps);
    void                destroy(VkDevice device, VmaAllocator vma);

    u64                 delta_time(u32 from, u32 to) const;
    u64                 max_time(u32 from, u32 to) const;

    void                reset(VkCommandBuffer cmd);
    void                record(VkCommandBuffer cmd, IsGraphics g = IsGraphics::NO);
    void                copy_results(VkCommandBuffer cmd) const;

private:
    VkQueryPool pool{};
    Buffer      buffer{};
    u32         query{};
    u32         max_queries{};
};


struct RecordedFrames {
    u32 mask{};
    bool did_record(u32 frame) const { return (mask & (1u << frame)) != 0; }
    void record(u32 frame) { mask &= ~(1u << frame); }
    void reset() { mask = UINT32_MAX; }
};


struct ResourceSync {
    ResourceSync() = default;

    void init(void* data, usize size, u32 max_buffers, u32 max_images);
    void clear();

    void read(VkCommandBuffer cmd, const BufferView& buffer, PipelineStage stage = PipelineStage::COMPUTE);
    void write(VkCommandBuffer cmd, const BufferView& buffer, PipelineStage stage = PipelineStage::COMPUTE);

    void read(VkCommandBuffer cmd, const Image& image, PipelineStage stage = PipelineStage::COMPUTE);
    void write(VkCommandBuffer cmd, const Image& image, PipelineStage stage = PipelineStage::COMPUTE);

    static constexpr usize allocation_size(u32 max_buffers, u32 max_images) { return max_buffers * sizeof(BufInfo) + max_images * sizeof(ImageInfo); }

private:
    struct BufInfo {
        BufferView    buffer{};
        u32           read: 1;
        u32           write: 1;
        u32           src_stage: 30;
    };

    struct ImageInfo {
        VkImage       image{};
        ImageFormat   format{};
        u32           read: 1;
        u32           write: 1;
        u32           usage: 4;
        u32           src_stage: 26;
    };

    void read_write(VkCommandBuffer cmd, const BufferView& buffer, PipelineStage stage, bool read, bool write);
    void read_write(VkCommandBuffer cmd, const Image& image, PipelineStage stage, bool read, bool write);
    bool insert(const BufInfo& info, BufInfo& prev);
    bool insert(const ImageInfo& info, ImageInfo& prev);
    void check_init() const;

    static void barrier(VkCommandBuffer cmd, const BufInfo& prev, const BufferView& buffer, PipelineStage stage, bool read, bool write);
    static void barrier(VkCommandBuffer cmd, const ImageInfo& prev, const Image& image, PipelineStage stage, bool read, bool write);

    BufInfo*    buf{};
    ImageInfo*  im{};
    u32         buffer_count{};
    u32         image_count{};
    u32         max_buffers{};
    u32         max_images{};
};

//endregion

//region Pipeline

struct ShaderMacro {
    const char* define{};
    const char* value{};
};


struct ShaderSrc {
    span<const char> src{};
    span<const ShaderMacro> macros{};
};


struct VertexAttribs {
    u32 formats[15]{};
    u32 count{};

    static VertexAttribs reflect(span<const u32> binary);
};


struct VertexBindings {
    u8 bind[16]{};
    u8 count() const { return bind[sizeof(bind) - 1]; }

    VertexBindings() { bind[sizeof(bind) - 1] = 1; }

    template<usize N>
    VertexBindings(const u32(&values)[N]) {
        for (u32 i = 0; i < N; ++i) bind[i] = i;
        bind[sizeof(bind)-1] = N;
    }

    template<usize N>
    VertexBindings(const u32(&ranges)[N][2]) {
        for (u32 i = 0; i < N; ++i)
            for (u32 j = ranges[i][0]; j <= ranges[i][1]; ++j)
                bind[j] = i;
        bind[sizeof(bind) - 1] = N;
    }
};


struct Pipeline {
    VkPipeline          vk{};
    operator VkPipeline() const { return vk; }

    struct Graphics;
    static Pipeline     create_compute(VkDevice device, VkPipelineLayout layout, span<const u32> bin);

    static Pipeline     create_compute(VkDevice device, VkPipelineLayout layout, ShaderSrc src);

    static Pipeline     create_graphics(VkDevice device, VkPipelineLayout layout, span<const u32> bin_vert, span<const u32> bin_frag, const Graphics& desc);

    static Pipeline     create_graphics(VkDevice device, VkPipelineLayout layout, ShaderSrc src_vert, ShaderSrc src_frag, const Graphics& desc);

    void                destroy(VkDevice device);

    struct Graphics {
        VkRenderPass    render_pass{};
        VertexBindings  bindings{};
        bool            depth{};
        u32             first_instance_attribute{-1u};
    };
};

//endregion

}


#endif //VDB_GPU_OBJECTS_H
