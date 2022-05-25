//
// Created by Djordje on 5/23/2022.
//

#include "Tree.h"
#include "../gpu/Context.h"
#include "../pvdb/pvdb_tree_write.h"

namespace pvdb {

//region tree

struct tree_gpu {
#ifdef PVDB_USE_IMAGES
    uint            levels;
#else
    uint            levels_channels_array;
#endif
    uint            log2dim_array;
    uint            total_log2dim_array[2];
    uint            atlas_log2dim_array;
    uint            data;
};
static_assert(sizeof(tree_gpu) == (PVDB_TREE_STRUCT_SIZE * sizeof(u32)), "tree_gpu does not match gpu version of pvdb_tree");


void Tree::init(gpu_context ctx, gpu_cmd cmd, const TreeDesc& desc, u32 array_index)
{
    index = array_index;
#ifdef PVDB_USE_IMAGES
    pvdb_tree_init(info, desc.levels, desc.log2dim, desc.leaf_count_log2dim);
#else
    pvdb_tree_init(info, desc.levels, desc.log2dim, desc.channels);
#endif

    auto& buffer = *new (storage) gpu::Buffer;
    buffer = ctx.create_buffer(desc.size * sizeof(u32), gpu::BufferType::STORAGE, gpu::BufferUsage::GPU);

#ifdef PVDB_USE_IMAGES
    auto& atlas = *new (atlas_storage) gpu::Image;
    const auto format = gpu::ImageFormat(u32(gpu::ImageFormat::R32_UINT) + PVDB_CHANNELS_LEAF - 1u);
    atlas = ctx.create_image(pvdb_atlas_dim_xy(info), pvdb_atlas_dim_xy(info), pvdb_atlas_dim_z(info), format, gpu::ImageUsage::STORAGE);
    transition_image_layout(cmd);
#endif

    const uint NodeSize[PVDB_MAX_LEVELS] {
        pvdb_node_size(info, 0),
        pvdb_node_size(info, 1),
        pvdb_node_size(info, 2)
    };

    pvdb_allocator_level alloc_levels[PVDB_ALLOCATOR_MAX_LEVELS]{};
#ifdef PVDB_USE_IMAGES
    pvdb_tree_init_allocator_levels(info, alloc_levels, desc.size, desc.leaf_count_log2dim);
#else
    pvdb_tree_init_allocator_levels(info, alloc_levels, desc.size, 0u);
#endif

    alloc.init(ctx, cmd, {alloc_levels, desc.levels - 1u}, pvdb_node_size(info, pvdb_root(info)));

    ctx.bind(PVDB_BINDING_TREE, gpu::BufferType::STORAGE, {(const gpu::Buffer*)storage, 1u}, array_index);
    alloc.bind(ctx, PVDB_BINDING_TREE_ALLOC, array_index);

#ifdef PVDB_USE_IMAGES
    ctx.bind(PVDB_BINDING_TREE_ATLAS, gpu::ImageLayout::GENERAL, {(const gpu::Image*)atlas_storage, 1u}, array_index);
#endif
}

void Tree::destroy(gpu_context ctx)
{
    alloc.destroy(ctx);
    ctx.destroy_buffer(*((gpu::Buffer*)storage));
#ifdef PVDB_USE_IMAGES
    ctx.destroy_image(*((gpu::Image*)atlas_storage));
#endif
}

void Tree::push_const(gpu_context ctx, gpu_cmd cmd, u32 offset) const
{
    tree_gpu t{};
#ifdef PVDB_USE_IMAGES
    t.levels = info.levels;
#else
    t.levels_channels_array = info.levels_channels_array;
#endif
    t.log2dim_array = info.log2dim_array;
    t.total_log2dim_array[0] = info.total_log2dim_array[0];
    t.total_log2dim_array[1] = info.total_log2dim_array[1];
    t.atlas_log2dim_array = info.atlas_log2dim_array;
    t.data = index;
    ctx.push_const<tree_gpu>(cmd, t, offset);
}

#ifdef PVDB_USE_IMAGES
void Tree::transition_image_layout(gpu_cmd cmd, bool write) const
{
    auto& atlas = *((const gpu::Image*)atlas_storage);
    auto dst_access = gpu::Access::SHADER_READ | (write ? gpu::Access::SHADER_WRITE : gpu::Access{});
    gpu::ImageBarrier b{
        gpu::Access{}       , dst_access,
        gpu::ImageLayout{}  , gpu::ImageLayout::GENERAL,
        atlas.vk,
        gpu::ImageFormat(atlas.format)
    };
    cmd.barrier(gpu::PipelineStage::TOP, gpu::PipelineStage::COMPUTE, {}, {}, {}, {&b, 1});
}
#endif

//endregion

//region trees

Trees::~Trees()
{
    assert(tree_slots.empty() && "Did not destroy all trees");
}

#ifdef PVDB_USE_IMAGES
void Trees::init(gpu_context ctx, u32 cl, u32 cn)
#else
void Trees::init(gpu_context ctx)
#endif
{
    p_ctx = &ctx;
#ifdef PVDB_USE_IMAGES
    channels_node = cl;
    channels_leaf = cl;
#endif
    ctx.bind(PVDB_BINDING_TREE, gpu::BufferType::STORAGE, {nullptr, PVDB_MAX_TREES});
    ctx.bind(PVDB_BINDING_TREE_ALLOC, gpu::BufferType::STORAGE, {nullptr, PVDB_MAX_TREES});

#ifdef PVDB_USE_IMAGES
    ctx.bind(PVDB_BINDING_TREE_ATLAS, gpu::ImageLayout::GENERAL, {nullptr, PVDB_MAX_TREES});
#endif
}

u32 Trees::create(gpu_cmd cmd, const TreeDesc& desc)
{
    const u32 index = tree_slots.alloc();
    tree_array[index].init(*p_ctx, cmd, desc, index);
    return index;
}

void Trees::destroy(u32 tree)
{
    tree_array[tree].destroy(*p_ctx);
    tree_slots.free(tree);
}

//endregion

}
