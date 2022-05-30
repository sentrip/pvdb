//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_CVDB_ALLOCATOR_H
#define PVDB_CVDB_ALLOCATOR_H

#include "../fwd.h"
#include "../../pvdb/util/pvdb_allocator.h"

namespace pvdb {


struct Allocator {
    Allocator() = default;

    void init(gpu_context ctx, gpu_cmd cmd, span<const pvdb_allocator_level> levels, u32 offset = 0, Device d = DEVICE_CPU);
    void destroy(gpu_context ctx);

    void bind(gpu_context ctx, u32 binding, u32 array_index) const;

private:
    u8 storage[GPU_BUFFER_SIZE_BYTES]{};
};

}

#endif //PVDB_CVDB_ALLOCATOR_H
