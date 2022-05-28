//
// Created by Djordje on 5/27/2022.
//

#include "Queue.h"
#include "../../gpu/Context.h"

namespace pvdb {

using namespace gpu;

void Queue::init(gpu_context ctx, gpu_cmd cmd, uint size, uint element_size, Device d)
{
    max_size = size;
    elem_size = element_size;

    auto& queue = *new (storage) Buffer;
    queue = ctx.create_buffer(sizeof(pvdb_queue_header) + size * element_size, BufferType::INDIRECT_STORAGE, d == DEVICE_GPU ? BufferUsage::GPU : BufferUsage::CPU);

    pvdb_queue_header q{};
    pvdb_queue_init(q, size);
    cmd.update_buffer(queue, 0, {(const u8*)&q, sizeof(pvdb_queue_header)});
    cmd.fill_buffer(queue, sizeof(pvdb_queue_header), queue.size - sizeof(pvdb_queue_header), 0);
}

void Queue::destroy(gpu_context ctx)
{
    ctx.destroy_buffer(*((Buffer*)storage));
}

void Queue::bind(gpu_context ctx, u32 binding, u32 array_index) const
{
    ctx.bind(binding, BufferType::STORAGE, {(const Buffer*)storage, 1u}, array_index);
}

void Queue::execute(gpu_cmd cmd) const
{
    auto& queue = *((const Buffer*)storage);

    // wait for all writes
    BufferBarrier to_indirect{Access::SHADER_WRITE, Access::INDIRECT_READ, {queue.vk, offsetof(pvdb_queue_header, indirect_x), 3u * sizeof(u32)}};
    cmd.barrier(PipelineStage::COMPUTE, PipelineStage::DRAW_INDIRECT, {}, {}, {&to_indirect, 1});

    // execute based on how many elements are in the queue
    cmd.dispatch_indirect(queue, offsetof(pvdb_queue_header, indirect_x));

    // wait for all reads
    BufferBarrier from_indirect{Access::INDIRECT_READ, Access::SHADER_WRITE, {queue.vk, offsetof(pvdb_queue_header, indirect_x), sizeof(u32)}};
    cmd.barrier(PipelineStage::DRAW_INDIRECT, PipelineStage::COMPUTE, {}, {}, {&from_indirect, 1});

    // reset queue
    set_count_indirect_x(cmd, 0u, 0u);
}

void Queue::push(const void *value, uint size, uint n) const
{
    assert(size == elem_size && "Size of pushed value does not match queue element size");
    auto& queue = *((const Buffer*)storage);
    auto& c = ((atom_t*)queue.data)[0];
    memcpy(queue.data + sizeof(pvdb_queue_header) + (c * size), value, n * size);
    c += n;
}

u32 Queue::fill(gpu_cmd cmd, const Queue& dst, u32 local_size, u32 offset) const
{
    assert(elem_size == dst.elem_size && "Queues must have matching element sizes in order to fill");

    auto& queue = *((const Buffer*)storage);
    auto& dst_queue = *((const Buffer*)dst.storage);

    const uint n = ((const atom_t*)queue.data)[0];
    const uint safe_n = n <= dst.max_size ? n : dst.max_size;

    // prepare dst queue for execution
    dst.set_count_indirect_x(cmd, safe_n, 1u + (safe_n-1u) / local_size);

    // copy safe_n elements from offset to dst queue
    cmd.copy_buffer(dst_queue.vk, sizeof(pvdb_queue_header), queue.vk, sizeof(pvdb_queue_header) + offset * elem_size, safe_n * elem_size);

    // wait for writes to dst queue
    BufferBarrier to_read{Access::TRANSFER_WRITE, Access::SHADER_READ, {dst_queue.vk, 0u, dst_queue.size}};
    cmd.barrier(PipelineStage::TRANSFER, PipelineStage::COMPUTE, {}, {}, {&to_read, 1u});

    // subtract safe_n elements from src queue
    ((atom_t*)queue.data)[0] = n - safe_n;

    // if we consumed all elements, return 0, otherwise return offset required for next call to fill
    return n == safe_n ? 0u : safe_n;
}

void Queue::set_count_indirect_x(gpu_cmd cmd, u32 count, u32 indirect_x) const
{
    auto& queue = *((const Buffer*)storage);
    cmd.fill_buffer(queue, offsetof(pvdb_queue_header, count), sizeof(u32), count);
    cmd.fill_buffer(queue, offsetof(pvdb_queue_header, indirect_x), sizeof(u32), indirect_x);
}


void Queues::init(gpu_context ctx, u32 binding, u32 array_size)
{
    p_ctx = &ctx;
    bind = binding;
    ctx.bind(binding, BufferType::STORAGE, {nullptr, array_size});
}

void Queues::init(gpu_cmd cmd, uint index, uint size, uint element_size, Device d)
{
    assert(p_ctx);
    queues[index].init(*p_ctx, cmd, size, element_size, d);
    queues[index].bind(*p_ctx, bind, index);
}

void Queues::destroy(uint index)
{
    assert(p_ctx);
    queues[index].destroy(*p_ctx);
}

void Queues::execute(gpu_cmd cmd, uint index) const
{
    queues[index].execute(cmd);
}

}
