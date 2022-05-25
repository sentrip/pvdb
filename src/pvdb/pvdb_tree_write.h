//
// Created by Djordje on 5/22/2022.
//

#ifndef PVDB_TREE_WRITE_H
#define PVDB_TREE_WRITE_H

#include "pvdb_tree_read.h"
#include "pvdb_allocator.h"

//region definitions

#ifdef PVDB_USE_IMAGES
#ifdef PVDB_C

#define pvdb_atlas_write(t, p, v)                   (pvdb_tree_data_atlas((t).data)[pvdb_atlas_begin(t, p)]) = v
#define pvdb_atlas_write_channel(t, p, v, c)        (pvdb_tree_data_atlas((t).data)[pvdb_atlas_begin(t, p) + ((c) * pvdb_dim3(t, 0))]) = v
PVDB_INLINE void pvdb_atlas_write_channels(pvdb_tree_in t, PVDB_IN(ivec3) p, const uvec4& v) {
    const uint leaf_dim3 = pvdb_dim3(t, 0);
    const uint offset = pvdb_atlas_begin(t, p);
    for (uint i = 0; i < PVDB_CHANNELS_LEAF; ++i)
        t.data.atlas[offset + i * leaf_dim3] = v[i];
}

#else

#define pvdb_atlas_write_channels(t, p, v)          imageStore(PVDB_GLOBAL_TREE_ATLAS[pvdb_tree_data_atlas((t).data)], p, v)
#define pvdb_atlas_write(t, p, v)                   pvdb_atlas_write_channel(t, p, v, 0)
PVDB_INLINE void pvdb_atlas_write_channel(pvdb_tree_in t, PVDB_IN(ivec3) p, uint v, uint channel) {
    uvec4 current = pvdb_atlas_read_channels(t, p);
    current[channel] = v;
    pvdb_atlas_write_channels(t, p, current);
}

#endif
#endif

PVDB_INLINE void
pvdb_tree_init_allocator_levels(
    pvdb_tree_in        tree,
    PVDB_ARRAY_INOUT(pvdb_allocator_level, alloc_levels, PVDB_ALLOCATOR_MAX_LEVELS),
    uint                size,
    uint                leaf_count_log2dim)
{
    const uint alloc_level_count = pvdb_levels(tree) - 1u;

#ifdef PVDB_USE_IMAGES
    const uint begin = 1u;
    const uint per_level = (size - pvdb_node_size(tree, pvdb_root(tree))) / alloc_level_count;
    alloc_levels[0].block_size = 1u;
    alloc_levels[0].max_allocations = 1u << leaf_count_log2dim;
    alloc_levels[0].data_offset = 1u;
#else
    const uint begin = 0u;
    const uint per_level = (size - pvdb_node_size(tree, pvdb_root(tree))) / alloc_level_count;
#endif
    for (uint i = begin; i < alloc_level_count; ++i) {
        alloc_levels[i].block_size = pvdb_node_size(tree, i);
        alloc_levels[i].max_allocations = per_level / alloc_levels[i].block_size;
#ifdef PVDB_ALLOCATOR_MASK
        alloc_levels[i].data_offset = -1u;
#else
        alloc_levels[i].data_offset = pvdb_node_size(tree, pvdb_root(tree));
#endif
    }
}


PVDB_INLINE void
pvdb_tree_init_allocator(
    pvdb_tree_in        tree,
    pvdb_buf_inout      alloc,
    uint                size,
    uint                leaf_count_log2dim)
{
    pvdb_allocator_level levels[PVDB_ALLOCATOR_MAX_LEVELS];
    pvdb_tree_init_allocator_levels(tree, levels, size, leaf_count_log2dim);
#ifdef PVDB_ALLOCATOR_MASK
    pvdb_allocator_init(alloc, pvdb_levels(tree) - 1u, levels, pvdb_node_size(tree, pvdb_root(tree)));
#else
    pvdb_allocator_init(alloc, pvdb_levels(tree) - 1u, levels, 0u);
#endif
}

//endregion

//region write primitive

/// write value to node
PVDB_INLINE void
pvdb_write_node(
    pvdb_tree_in        tree,
    uint                level,
    uint                node,
    uint                index,
    uint                value)
{
    const uint address = node + pvdb_data_offset(tree, level, index);
    pvdb_tree_at(tree, address) = value;
}


/// write value to node at given channel
PVDB_INLINE void
pvdb_write_node(
    pvdb_tree_in        tree,
    uint                level,
    uint                node,
    uint                index,
    uint                value,
    uint                channel)
{
    const uint address = node + pvdb_data_offset_channel(tree, level, index, channel);
    pvdb_tree_at(tree, address) = value;
}


/// write node mask atomically
PVDB_INLINE bool
pvdb_write_node_mask(
    pvdb_tree_in        tree,
    uint                node,
    uint                index,
    bool                on)
{
    const uint address = node + pvdb_mask_offset(index);
    const uint mask = 1u << pvdb_mask_bit_i(index);
    const uint current = pvdb_tree_at(tree, address);
    const uint updated = on ? (current | mask) : (current & ~mask);
    const uint prev = atomicCompSwap(pvdb_tree_at(tree, address), current, updated);
    return prev == current && (current != updated);
}


/// write value to the first channel in the given leaf at the given position
PVDB_INLINE void
pvdb_write_leaf(
    pvdb_tree_in        tree,
    uint                leaf,
    PVDB_IN(ivec3)      p,
    uint                v)
{
#ifdef PVDB_USE_IMAGES
    const ivec3 a = pvdb_atlas_index_to_offset(tree, leaf) + p;
    pvdb_atlas_write(tree, a, v);
#else
    const uint i = pvdb_coord_local_to_index(tree, p, 0);
    pvdb_tree_at(tree, leaf + pvdb_data_offset(tree, 0, i)) = v;
#endif
}


/// write value to the given channel in the given leaf at the given position
PVDB_INLINE void
pvdb_write_leaf(
    pvdb_tree_in        tree,
    uint                leaf,
    PVDB_IN(ivec3)      p,
    uint                v,
    uint                channel)
{
#ifdef PVDB_USE_IMAGES
    const ivec3 a = pvdb_atlas_index_to_offset(tree, leaf) + p;
    pvdb_atlas_write_channel(tree, a, v, channel);
#else
    const uint i = pvdb_coord_local_to_index(tree, p, 0);
    pvdb_tree_at(tree, leaf + pvdb_data_offset_channel(tree, 0, i, channel)) = v;
#endif
}

#ifdef PVDB_USE_IMAGES
/// write value to all channels in the given leaf at the given position
PVDB_INLINE void
pvdb_write_leaf(
    pvdb_tree_in        tree,
    uint                leaf,
    PVDB_IN(ivec3)      p,
    uvec4               v)
{
    const ivec3 a = pvdb_atlas_index_to_offset(tree, leaf) + p;
    pvdb_atlas_write_channels(tree, a, v);
}
#endif

//endregion

//region insert

/// insert node/tile
PVDB_INLINE uint
pvdb_try_insert_node(
    pvdb_tree_in        tree,
    uint                child_level,
    uint                parent,
    uint                child_index_in_parent,
    uint                child)
{
    const uint parent_level = child_level + 1;
    const uint address_data = parent + pvdb_data_offset(tree, parent_level, child_index_in_parent);
    uint node = pvdb_tree_at(tree, address_data);
    while (node == 0u) {
        // If we are the first thread to set the bit, insert the new node
        if (pvdb_write_node_mask(tree, parent, child_index_in_parent, true)) {
//            PVDB_PRINTF("\n\tWRITE MASK: node: %u, index: %u\n", parent, child_index_in_parent);
            if (!pvdb_is_tile(child)) {
                child = pvdb_allocator_alloc(pvdb_tree_data_alloc(tree.data), child_level);
                if (child_level > 0u) {
                    pvdb_tree_at(tree, child + 0u) = parent;
                    pvdb_tree_at(tree, child + 1u) = child_index_in_parent;
                }
            }

            pvdb_write_node(tree, parent_level, parent, child_index_in_parent, child);
            node = child;
            if (parent_level > 1)
            PVDB_PRINTF("\n\tINSERT: parent_level: %u, parent: %u, node: %u, index: %u\n", parent_level, parent, node, child_index_in_parent);
        }
        // Else wait for the other thread to insert the node
        else {
            node = pvdb_tree_at(tree, address_data);
        }
    }
    return node;
}


/// insert node/tile at any level
PVDB_INLINE uint
pvdb_insert(
    pvdb_tree_in        tree,
    uint                child_level,
    PVDB_IN(ivec3)      p,
    uint                child)
{
    uint node = PVDB_ROOT_NODE;
    uint level = pvdb_root(tree) - 1;
    for (;;) {
        const ivec3 local = pvdb_coord_global_to_local(tree, p, level + 1);
        const uint index_in_parent = pvdb_coord_local_to_index(tree, local, level + 1);
        node = pvdb_try_insert_node(tree, level, node, index_in_parent, (level == child_level) ? child : 0u);
        if (pvdb_is_tile(node) || level-- == child_level) break;
    }
    return node;
}

//endregion

//region set/replace

/// set value in the first channel at the given coord (inserts new nodes if required)
PVDB_INLINE void
pvdb_set(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uint                v)
{
    const uint leaf = pvdb_insert(tree, 0, p, 0);
    if (pvdb_is_tile(leaf)) return;
//    PVDB_PRINTF("\n\t(%d, %d, %d): leaf: %u\n", p.x, p.y, p.z, leaf);
    pvdb_write_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0), v);
}


/// set value in the given channel at the given coord (inserts new nodes if required)
PVDB_INLINE void
pvdb_set(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uint                v,
    uint                channel)
{
    const uint leaf = pvdb_insert(tree, 0, p, 0);
    if (pvdb_is_tile(leaf)) return;
    pvdb_write_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0), v, channel);
}

#ifdef PVDB_USE_IMAGES
/// set value in all channels at the given coord (inserts new nodes if required)
PVDB_INLINE void
pvdb_set(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uvec4               v)
{
    const uint leaf = pvdb_insert(tree, 0, p, 0);
    if (pvdb_is_tile(leaf)) return;
    pvdb_write_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0), v);
}
#endif


/// set value in the first channel at the given coord (does not attempt to insert new nodes)
PVDB_INLINE void
pvdb_replace(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uint                v)
{
    const uint leaf = pvdb_traverse_at_least(tree, 0, p);
    if (leaf == 0u || pvdb_is_tile(leaf)) return;
    pvdb_write_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0), v);
}


/// set value in the given channel at the given coord (does not attempt to insert new nodes)
PVDB_INLINE void
pvdb_replace(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uint                v,
    uint                channel)
{
    const uint leaf = pvdb_traverse_at_least(tree, 0, p);
    if (leaf == 0u || pvdb_is_tile(leaf)) return;
    pvdb_write_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0), v, channel);
}


#ifdef PVDB_USE_IMAGES
/// set value in all channels at the given coord (does not attempt to insert new nodes)
PVDB_INLINE void
pvdb_replace(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uvec4               v)
{
    const uint leaf = pvdb_traverse_at_least(tree, 0, p);
    if (leaf == 0u || pvdb_is_tile(leaf)) return;
    pvdb_write_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0), v);
}
#endif


//endregion

#endif //PVDB_TREE_WRITE_H
