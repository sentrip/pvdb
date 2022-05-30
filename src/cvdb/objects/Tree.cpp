//
// Created by Djordje on 5/23/2022.
//

#include "Tree.h"
#include "../../gpu/Context.h"
#include "../../pvdb/tree/pvdb_tree_write.h"

namespace pvdb {

//region tree

struct tree_gpu {
    uint            levels_channels_array;
    uint            log2dim_array;
    uint            total_log2dim_array[2];
    uint            data;
};
static_assert(sizeof(tree_gpu) == (PVDB_TREE_STRUCT_SIZE * sizeof(u32)), "tree_gpu does not match gpu version of pvdb_tree");


void Tree::init(gpu_context ctx, gpu_cmd cmd, const TreeDesc& desc, u32 array_index, Device d)
{
    index = array_index;
    pvdb_tree_init(info, desc.levels, desc.log2dim, desc.channels);

    auto& buffer = *new (storage) gpu::Buffer;
    buffer = ctx.create_buffer(desc.size * sizeof(u32), gpu::BufferType::STORAGE, d == DEVICE_GPU ? gpu::BufferUsage::GPU : gpu::BufferUsage::CPU);

    pvdb_allocator_level alloc_levels[PVDB_ALLOCATOR_MAX_LEVELS]{};
    pvdb_tree_init_allocator_levels(info, desc.size, alloc_levels);
    alloc.init(ctx, cmd, {alloc_levels, desc.levels - 1u}, pvdb_node_size(info, pvdb_root(info)));

    ctx.bind(PVDB_BINDING_TREE, gpu::BufferType::STORAGE, {(const gpu::Buffer*)storage, 1u}, array_index);
    alloc.bind(ctx, PVDB_BINDING_TREE_ALLOC, array_index);
}

void Tree::destroy(gpu_context ctx)
{
    alloc.destroy(ctx);
    ctx.destroy_buffer(*((gpu::Buffer*)storage));
}

void Tree::push_const(gpu_context ctx, gpu_cmd cmd, u32 offset) const
{
    tree_gpu t{};
    t.levels_channels_array = info.levels_channels_array;
    t.log2dim_array = info.log2dim_array;
    t.total_log2dim_array[0] = info.total_log2dim_array[0];
    t.total_log2dim_array[1] = info.total_log2dim_array[1];
    t.data = index;
    ctx.push_const<tree_gpu>(cmd, t, offset);
}

//endregion

//region trees

Trees::~Trees()
{
    assert(tree_slots.empty() && "Did not destroy all trees");
}

void Trees::init(gpu_context ctx)
{
    p_ctx = &ctx;
    ctx.bind(PVDB_BINDING_TREE, gpu::BufferType::STORAGE, {nullptr, PVDB_MAX_TREES});
    ctx.bind(PVDB_BINDING_TREE_ALLOC, gpu::BufferType::STORAGE, {nullptr, PVDB_MAX_TREES});
}

u32 Trees::create(gpu_cmd cmd, const TreeDesc& desc, Device d)
{
    const u32 index = tree_slots.alloc();
    tree_array[index].init(*p_ctx, cmd, desc, index, d);
    return index;
}

void Trees::destroy(u32 tree)
{
    tree_array[tree].destroy(*p_ctx);
    tree_slots.free(tree);
}

//endregion

}
