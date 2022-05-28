//
// Created by Djordje on 5/27/2022.
//

#ifndef PVDB_QUEUE_H
#define PVDB_QUEUE_H

#include "../fwd.h"

#define PVDB_QUEUE_HEADER_ONLY
#include "../../pvdb/pvdb_queue.h"


namespace pvdb {


struct Queue {
    Queue() = default;

    template<typename T>
    void init(gpu_context ctx, gpu_cmd cmd, uint size, Device d = DEVICE_GPU) { init(ctx, cmd, size, sizeof(T), d); }

    void init(gpu_context ctx, gpu_cmd cmd, uint size, uint element_size, Device d);
    void destroy(gpu_context ctx);

    void bind(gpu_context ctx, u32 binding, u32 array_index) const;
    void execute(gpu_cmd cmd) const;

    u32  fill(gpu_cmd cmd, const Queue& dst, u32 local_size, u32 offset = 0) const;

    template<typename T>
    void push(const T& value) const { push(&value, sizeof(T), 1u); }

    template<typename T>
    void push(span<const T> values) const { push(values.data(), sizeof(T), values.size()); }

private:
    void push(const void* value, uint size, uint n) const;
    void set_count_indirect_x(gpu_cmd cmd, u32 count, u32 indirect_x) const;

    u8 storage[GPU_BUFFER_SIZE_BYTES]{};
    u32 max_size{};
    u32 elem_size{};
};


struct Queues {
    Queues() = default;

    void init(gpu_context ctx, u32 binding, u32 array_size = 32);

    template<typename T>
    void init(gpu_cmd cmd, uint index, uint size, Device d = DEVICE_GPU) { init(cmd, index, size, sizeof(T), d); }

    void init(gpu_cmd cmd, uint index, uint size, uint element_size, Device d = DEVICE_GPU);
    void destroy(uint index);

    void execute(gpu_cmd cmd, uint index) const;

    template<typename T>
    void push(u32 index, const T& value) const { queues[index].push(value); }

    template<typename T>
    void push(u32 index, span<const T> values) const { queues[index].push(values); }

    u32  fill(gpu_cmd cmd, u32 dst_index, u32 src_index, u32 local_size, u32 offset = 0) const { return queues[src_index].fill(cmd, queues[dst_index], local_size, offset); }

private:
    Queue queues[32]{};
    const gpu::Context* p_ctx{};
    u32 bind{};
};

}

#endif //PVDB_QUEUE_H
