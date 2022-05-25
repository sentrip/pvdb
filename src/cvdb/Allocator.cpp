//
// Created by Djordje on 5/23/2022.
//

#include "Allocator.h"
#include "../gpu/Context.h"

namespace pvdb {

void Allocator::init(gpu_context ctx, gpu_cmd cmd, span<const pvdb_allocator_level> levels, u32 offset)
{
    uint max_allocations[PVDB_ALLOCATOR_MAX_LEVELS];
    pvdb_allocator_level levels_array[PVDB_ALLOCATOR_MAX_LEVELS];
    memcpy(levels_array, levels.data(), levels.size() * sizeof(pvdb_allocator_level));
    for (u32 i = 0; i < levels.size(); ++i) max_allocations[i] = levels[i].max_allocations;

    pvdb_buf_t<PVDB_ALLOCATOR_MAX_LEVELS*sizeof(pvdb_allocator)> alloc{};
    const u32 alloc_size = pvdb_allocator_init(alloc, levels.size(), levels_array, offset);
    const u32 alloc_buffer_size = pvdb_allocator_size(levels.size(), max_allocations);

    auto& buffer = *new (storage) gpu::Buffer;
    buffer = ctx.create_buffer(alloc_buffer_size * sizeof(u32), gpu::BufferType::STORAGE, gpu::BufferUsage::GPU);

    const u32 offset_bytes = alloc_size * sizeof(u32);
    cmd.update_buffer(buffer, 0, {(const u8*)alloc.data, offset_bytes});
    cmd.fill_buffer(buffer, offset_bytes, buffer.size - offset_bytes, 0);
}

void Allocator::destroy(gpu_context ctx)
{
    ctx.destroy_buffer(*((gpu::Buffer*)storage));
}

void Allocator::bind(gpu_context ctx, u32 binding, u32 array_index) const
{
    ctx.bind(binding, gpu::BufferType::STORAGE, {(const gpu::Buffer*)storage, 1}, array_index);
}

}