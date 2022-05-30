//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_CVDB_TREE_H
#define PVDB_CVDB_TREE_H

#include "Allocator.h"
#include "../../pvdb/tree/pvdb_tree.h"

namespace pvdb {


struct TreeDesc {
    u32 size{};
    u32 levels{};
    u32 log2dim[PVDB_MAX_LEVELS]{};
    u32 channels[PVDB_MAX_LEVELS]{1,1,1,1,1,1,1,1};
};


struct Tree {
    Tree() = default;

    void init(gpu_context ctx, gpu_cmd cmd, const TreeDesc& desc, u32 array_index = 0, Device d = DEVICE_CPU);
    void destroy(gpu_context ctx);

    void push_const(gpu_context ctx, gpu_cmd cmd, u32 offset = 0) const;

    operator const pvdb_tree&() const { return info; }

private:
    pvdb_tree info{};
    u8        storage[GPU_BUFFER_SIZE_BYTES]{};
    Allocator alloc{};
    u32       index{};
};


struct Trees {
    Trees() = default;
    ~Trees();

    void init(gpu_context ctx);
    u32  create(gpu_cmd cmd, const TreeDesc& desc, Device d = DEVICE_CPU);
    void destroy(u32 tree);

    void push_const(gpu_cmd cmd, u32 tree, u32 offset = 0) const { tree_array[tree].push_const(*p_ctx, cmd, offset); }
    const Tree& operator[](u32 t) const { return tree_array[t]; }

private:
    slots<PVDB_MAX_TREES>   tree_slots{};
    Tree                    tree_array[PVDB_MAX_TREES]{};
    const gpu::Context*     p_ctx{};
};

}

#endif //PVDB_CVDB_TREE_H
