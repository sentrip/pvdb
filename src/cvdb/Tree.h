//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_CVDB_TREE_H
#define PVDB_CVDB_TREE_H

#include "Allocator.h"
#include "../pvdb/pvdb_tree.h"

namespace pvdb {


struct TreeDesc {
    u32 size{};
    u32 levels{};
    u32 log2dim[PVDB_MAX_LEVELS]{};
#ifdef PVDB_USE_IMAGES
    u32 leaf_count_log2dim{};
#else
    u32 channels[PVDB_MAX_LEVELS]{1,1,1,1,1,1,1,1};
#endif
};


struct Tree {
    Tree() = default;

    void init(gpu_context ctx, gpu_cmd cmd, const TreeDesc& desc, u32 array_index = 0);
    void destroy(gpu_context ctx);

    void push_const(gpu_context ctx, gpu_cmd cmd, u32 offset = 0) const;
#ifdef PVDB_USE_IMAGES
    void transition_image_layout(gpu_cmd cmd, bool write = true) const;
#endif

private:
    pvdb_tree info{};
    u8        storage[GPU_BUFFER_SIZE_BYTES]{};
    Allocator alloc{};
#ifdef PVDB_USE_IMAGES
    u8        atlas_storage[GPU_IMAGE_SIZE_BYTES]{};
#endif
    u32       index{};
};


struct Trees {
    Trees() = default;
    ~Trees();
#ifdef PVDB_USE_IMAGES
    void init(gpu_context ctx, u32 channels_leaf = 1u, u32 channels_node = 1u);
#else
    void init(gpu_context ctx);
#endif
    u32  create(gpu_cmd cmd, const TreeDesc& desc);
    void destroy(u32 tree);

    const Tree& operator[](u32 t) const { return tree_array[t]; }

private:
    slots<PVDB_MAX_TREES>   tree_slots{};
    Tree                    tree_array[PVDB_MAX_TREES]{};
    const gpu::Context*     p_ctx{};
#ifdef PVDB_USE_IMAGES
    u32                     channels_leaf{}, channels_node{};
#endif
};

}

#endif //PVDB_CVDB_TREE_H
