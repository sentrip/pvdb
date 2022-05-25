//
// Created by Djordje on 5/21/2022.
//

#ifndef PVDB_TREE_READ_H
#define PVDB_TREE_READ_H

#include "pvdb_tree.h"

//region definitions

#ifndef PVDB_GLOBAL_TREE
#define PVDB_GLOBAL_TREE                    GlobalTree
#endif

#ifndef PVDB_GLOBAL_TREE_ATLAS
#define PVDB_GLOBAL_TREE_ATLAS              GlobalTreeAtlas
#endif

#define pvdb_tree_at(tree, addr)                    pvdb_buf_at(pvdb_tree_data_nodes((tree).data), PVDB_GLOBAL_TREE, addr)

#ifdef PVDB_USE_IMAGES
#ifdef PVDB_C

#define pvdb_atlas_begin(t, p)                      ((pvdb_atlas_offset_to_index(t, p) * pvdb_leaf_size(t)) + pvdb_coord_local_to_index(t, p, 0))
#define pvdb_atlas_read(t, p)                       (pvdb_tree_data_atlas((t).data)[pvdb_atlas_begin(t, p)])
#define pvdb_atlas_read_channel(t, p, c)            (pvdb_tree_data_atlas((t).data)[pvdb_atlas_begin(t, p) + ((c) * pvdb_dim3(t, 0))])
static inline uvec4 pvdb_atlas_read_channels(pvdb_tree_in t, PVDB_IN(ivec3) p) {
    uvec4 r{};
    const uint leaf_dim3 = pvdb_dim3(t, 0);
    const uint offset = pvdb_atlas_begin(t, p);
    for (uint i = 0; i < PVDB_CHANNELS_LEAF; ++i) r[i] = t.data.atlas[offset + i * leaf_dim3];
    return r;
}

#else

#define pvdb_atlas_read(t, p)                       imageLoad(PVDB_GLOBAL_TREE_ATLAS[pvdb_tree_data_atlas((t).data)], p).x
#define pvdb_atlas_read_channel(t, p, c)            imageLoad(PVDB_GLOBAL_TREE_ATLAS[pvdb_tree_data_atlas((t).data)], p)[c]
#define pvdb_atlas_read_channels(t, p)              imageLoad(PVDB_GLOBAL_TREE_ATLAS[pvdb_tree_data_atlas((t).data)], p)

#endif
#endif

//endregion

//region read primitive

/// read value from address
PVDB_INLINE uint
pvdb_read_address(
    pvdb_tree_in        tree,
    uint                address)
{
    return pvdb_tree_at(tree, address);
}


/// read bit from address
PVDB_INLINE bool
pvdb_read_bit(
    pvdb_tree_in        tree,
    uint                address,
    uint                index)
{
    PVDB_ASSERT(index < 32 && "Index out of range");
    const uint mask = 1u << pvdb_mask_bit_i(index);
    return (pvdb_tree_at(tree, address) & mask) != 0u;
}


/// read value from node
PVDB_INLINE uint
pvdb_read_node(
    pvdb_tree_in        tree,
    uint                level,
    uint                node,
    uint                index)
{
    const uint address = node + pvdb_data_offset(tree, level, index);
    return pvdb_tree_at(tree, address);
}


/// read value from node at given channel
PVDB_INLINE uint
pvdb_read_node(
    pvdb_tree_in        tree,
    uint                level,
    uint                node,
    uint                index,
    uint                channel)
{
    return pvdb_tree_at(tree, node + pvdb_data_offset_channel(tree, level, index, channel));
}


/// read node mask
PVDB_INLINE bool
pvdb_read_node_mask(
    pvdb_tree_in        tree,
    uint                node,
    uint                index)
{
    return pvdb_read_bit(tree, node + pvdb_mask_offset(index), index & 31u);
}


/// read value from leaf
PVDB_INLINE uint
pvdb_read_leaf(
    pvdb_tree_in        tree,
    uint                leaf,
    PVDB_IN(ivec3)      p)
{
#ifdef PVDB_USE_IMAGES
    const ivec3 a = pvdb_atlas_index_to_offset(tree, leaf) + p;
    return pvdb_atlas_read(tree, a);
#else
    const uint i = pvdb_coord_local_to_index(tree, p, 0);
    return pvdb_tree_at(tree, leaf + pvdb_data_offset(tree, 0, i));
#endif
}


/// read value from leaf at given channel
PVDB_INLINE uint
pvdb_read_leaf(
    pvdb_tree_in        tree,
    uint                leaf,
    PVDB_IN(ivec3)      p,
    uint                channel)
{
#ifdef PVDB_USE_IMAGES
    const ivec3 a = pvdb_atlas_index_to_offset(tree, leaf) + p;
    return pvdb_atlas_read_channel(tree, a, channel);
#else
    const uint i = pvdb_coord_local_to_index(tree, p, 0);
    return pvdb_tree_at(tree, leaf + pvdb_data_offset_channel(tree, 0, i, channel));
#endif
}

#ifdef PVDB_USE_IMAGES
/// read value from leaf in all channels
PVDB_INLINE uvec4
pvdb_read_leaf_channels(
    pvdb_tree_in        tree,
    uint                leaf,
    PVDB_IN(ivec3)      p)
{
    const ivec3 a = pvdb_atlas_index_to_offset(tree, leaf) + p;
    return pvdb_atlas_read_channels(tree, a);
}
#endif

//endregion

//region node header

/// get the parent node of the given node
PVDB_INLINE uint
pvdb_get_parent(
    pvdb_tree_in        tree,
    uint                node)
{
    return pvdb_tree_at(tree, node + 0u);
}


/// get the index in the parent node of the given node
PVDB_INLINE uint
pvdb_get_index_in_parent(
    pvdb_tree_in        tree,
    uint                node)
{
    return pvdb_tree_at(tree, node + 1u);
}

/// get entire node header from the given node
PVDB_INLINE pvdb_node_header
pvdb_get_node(
    pvdb_tree_in        tree,
    uint                node)
{
    pvdb_node_header n;
    n.parent = pvdb_tree_at(tree, node + 0u);
    n.index_in_parent = pvdb_tree_at(tree, node + 1u);
    n.mesh = 0u;
    return n;
}

//endregion

//region traverse

/// traverse from starting level and record traversed level
PVDB_INLINE uint
pvdb_traverse(
    pvdb_tree_in        tree,
    uint                node,
    uint                start_level,
    PVDB_INOUT(uint)    target_level,
    PVDB_IN(ivec3)      p)
{
    const uint target = target_level;
    target_level = start_level;
    for (;;) {
        const uint next = pvdb_read_node(tree, target_level, node, pvdb_coord_global_to_index(tree, p, target_level));
        if (next == PVDB_ROOT_NODE) return node;
        if (--target_level == target || pvdb_is_tile(next)) return next;
        node = next;
    }
    return node;
}


/// traverse and record traversed level
PVDB_INLINE uint
pvdb_traverse(
    pvdb_tree_in        tree,
    PVDB_INOUT(uint)    level,
    PVDB_IN(ivec3)      p)
{
    return pvdb_traverse(tree, PVDB_ROOT_NODE, pvdb_root(tree), level, p);
}


/// traverse and record traversed level
PVDB_INLINE uint
pvdb_traverse_at_least(
    pvdb_tree_in        tree,
    uint                level,
    PVDB_IN(ivec3)      p)
{
    const uint l = level;
    const uint node = pvdb_traverse(tree, level, p);
    if (level != l && !pvdb_is_tile(node)) return PVDB_ROOT_NODE;
    return node;
}

//endregion

//region get

/// leaf value at given global coord
PVDB_INLINE uint
pvdb_get(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p)
{
    const uint leaf = pvdb_traverse_at_least(tree, 0u, p);
    if (leaf == PVDB_ROOT_NODE || pvdb_is_tile(leaf)) return leaf & PVDB_NODE;
    return pvdb_read_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0u));
}


/// leaf value at given global coord in the given channel
PVDB_INLINE uint
pvdb_get(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p,
    uint                channel)
{
    const uint leaf = pvdb_traverse_at_least(tree, 0u, p);
    if (leaf == PVDB_ROOT_NODE || pvdb_is_tile(leaf)) return leaf & PVDB_NODE;
    return pvdb_read_leaf(tree, leaf, pvdb_coord_global_to_local(tree, p, 0u), channel);
}


#ifdef PVDB_USE_IMAGES
/// leaf value at given global coord in all channels
PVDB_INLINE uvec4
pvdb_get_channels(
    pvdb_tree_in        tree,
    PVDB_IN(ivec3)      p)
{
    const uint leaf = pvdb_traverse_at_least(tree, 0u, p);
    if (leaf == PVDB_ROOT_NODE || pvdb_is_tile(leaf)) return uvec4(leaf & PVDB_NODE, 0, 0, 0);
    return pvdb_read_leaf_channels(tree, leaf, pvdb_coord_global_to_local(tree, p, 0u));
}
#endif

//endregion

#endif //PVDB_TREE_READ_H
