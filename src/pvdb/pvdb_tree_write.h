//
// Created by Djordje on 5/22/2022.
//

#ifndef PVDB_TREE_WRITE_H
#define PVDB_TREE_WRITE_H

#include "pvdb_tree_read.h"
#include "pvdb_allocator.h"

//region definitions

PVDB_INLINE void
pvdb_tree_init_allocator_levels(
    pvdb_tree_in        tree,
    uint                size,
    PVDB_ARRAY_INOUT(pvdb_allocator_level, alloc_levels, PVDB_ALLOCATOR_MAX_LEVELS))
{
    const uint alloc_level_count = pvdb_levels(tree) - 1u;
    const uint per_level = (size - pvdb_node_size(tree, pvdb_root(tree))) / alloc_level_count;
    for (uint i = 0u; i < alloc_level_count; ++i) {
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
    uint                size)
{
    pvdb_allocator_level levels[PVDB_ALLOCATOR_MAX_LEVELS];
    pvdb_tree_init_allocator_levels(tree, size, levels);
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
    const uint i = pvdb_coord_local_to_index(tree, p, 0);
    pvdb_tree_at(tree, leaf + pvdb_data_offset(tree, 0, i)) = v;
    atomicAnd(pvdb_tree_at(tree, leaf + pvdb_mask_offset(i)), ~(1u << pvdb_mask_bit_i(i)));
    atomicOr(pvdb_tree_at(tree, leaf + pvdb_mask_offset(i)), uint(v != 0u) << pvdb_mask_bit_i(i));
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
    const uint i = pvdb_coord_local_to_index(tree, p, 0);
    pvdb_tree_at(tree, leaf + pvdb_data_offset_channel(tree, 0, i, channel)) = v;
}

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
            if (child == 0u) {
                child = pvdb_allocator_alloc(pvdb_tree_data_alloc(tree.data), child_level);
            }
            if (!pvdb_is_tile(child)) {
                pvdb_tree_at(tree, child + 0u) = parent;
                pvdb_tree_at(tree, child + 1u) = child_index_in_parent;
            }
            pvdb_tree_at(tree, address_data) = child;
            node = child;
//            PVDB_PRINTF("\n\tINSERT: parent_level: %u, parent: %u, node: %u, index: %u\n", parent_level, parent, node, child_index_in_parent);
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
//    PVDB_PRINTF("INSERT global(%4d, %4d, %4d)\n", p.x, p.y, p.z);
    for (;;) {
        const ivec3 local = pvdb_coord_global_to_local(tree, p, level + 1);
        const uint index_in_parent = pvdb_coord_local_to_index(tree, local, level + 1);
//        uint parent = node;
        node = pvdb_try_insert_node(tree, level, node, index_in_parent, (level == child_level) ? child : 0u);
//        PVDB_PRINTF("INSERT global(%4d, %4d, %4d), local(%4d, %4d, %4d), level: %u, parent: %8u, node: %8u, index: %8u\n", p.x, p.y, p.z, local.x, local.y, local.z, level, parent, node, index_in_parent);
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

//endregion

//region copy

/// copy mask and value from src node and index at src level to dst node and index at dst lvl
PVDB_INLINE void
pvdb_copy_value(
    pvdb_tree_in        dst,
    uint                dst_level,
    uint                dst_node,
    uint                dst_index,
    pvdb_tree_in        src,
    uint                src_level,
    uint                src_node,
    uint                src_index)
{
    const bool is_on = pvdb_read_node_mask(src, src_node, src_index);
    const uint value = pvdb_read_node(src, src_level, src_node, src_index);
    pvdb_write_node_mask(dst, dst_node, dst_index, is_on);
    pvdb_write_node(dst, dst_level, dst_node, dst_index, value);
}


/// copy mask and value from src node and index at src level and channel to dst node and index at dst lvl and channel
PVDB_INLINE void
pvdb_copy_value(
    pvdb_tree_in        dst,
    uint                dst_level,
    uint                dst_node,
    uint                dst_index,
    uint                dst_channel,
    pvdb_tree_in        src,
    uint                src_level,
    uint                src_node,
    uint                src_index,
    uint                src_channel)
{
    const bool is_on = pvdb_read_node_mask(src, src_node, src_index);
    const uint value = pvdb_read_node(src, src_level, src_node, src_index, src_channel);
    pvdb_write_node_mask(dst, dst_node, dst_index, is_on);
    pvdb_write_node(dst, dst_level, dst_node, dst_index, value, dst_channel);
}

//endregion

#endif //PVDB_TREE_WRITE_H
