//
// Created by Djordje on 5/21/2022.
//

/** MACRO INPUTS
    * PVDB_GLOBAL_TREE          - name of global array (only required in glsl)
*/

#ifndef PVDB_TREE_H
#define PVDB_TREE_H

#include "pvdb_buffer.h"

//region definitions

/// globals
#define PVDB_MAX_TREES                  8
#define PVDB_MAX_LEVELS                 8
#define PVDB_TREE_STRUCT_SIZE           5 // levels, log2dim_array, total_log2dim[2], data
#define PVDB_HEADER_SIZE                4 // parent, index_in_parent, mesh, UNUSED
#define PVDB_ROOT_NODE                  0u
#define PVDB_TILE                       0x80000000u
#define PVDB_NODE                       0x7fffffffu

/// tree dimension
#define PVDB_LOG2DIM_MASK               0xfu
#define PVDB_TOTAL_LOG2DIM_MASK         0xffu
#define PVDB_LOG2DIM_SHIFT              2u
#define PVDB_TOTAL_LOG2DIM_SHIFT        3u
#define PVDB_CHANNELS_MASK              0x3u
#define PVDB_CHANNELS_SHIFT             1u
#define PVDB_LEVELS_MASK                0xfu
#define PVDB_LEVELS_SHIFT               8u


#ifdef PVDB_C
struct pvdb_tree;
struct pvdb_access;
typedef PVDB_IN(pvdb_tree)              pvdb_tree_in;
typedef PVDB_OUT(pvdb_tree)             pvdb_tree_out;
typedef PVDB_INOUT(pvdb_tree)           pvdb_tree_inout;
typedef PVDB_IN(pvdb_access)            pvdb_access_in;
typedef PVDB_OUT(pvdb_access)           pvdb_access_out;
typedef PVDB_INOUT(pvdb_access)         pvdb_access_inout;

struct pvdb_tree_data                   { atom_t* nodes{}; atom_t* alloc{}; };
#define pvdb_tree_data_nodes(d)         (d).nodes
#define pvdb_tree_data_alloc(d)         (d).alloc

#else
#define pvdb_tree_in                    PVDB_IN(pvdb_tree)
#define pvdb_tree_out                   PVDB_OUT(pvdb_tree)
#define pvdb_tree_inout                 PVDB_INOUT(pvdb_tree)
#define pvdb_access_in                  PVDB_IN(pvdb_access)
#define pvdb_access_out                 PVDB_OUT(pvdb_access)
#define pvdb_access_inout               PVDB_INOUT(pvdb_access)
#define pvdb_tree_data                  uint
#define pvdb_tree_data_nodes(d)         d
#define pvdb_tree_data_alloc(d)         d

#endif

//endregion

//region tree

/// tree
struct pvdb_tree {
    uint            levels_channels_array;
    uint            log2dim_array;
    uint            total_log2dim_array[2];
    pvdb_tree_data  data;
};


struct pvdb_node_header {
    uint  parent;
    uint  index_in_parent;
    uint  mesh;
    uint  PAD;
};


/// dimension math
#define pvdb_levels(t)                             ((t).levels_channels_array & PVDB_LEVELS_MASK)
#define pvdb_channels(t, lvl)                      ( ( (t).levels_channels_array >> (PVDB_LEVELS_SHIFT + ((lvl)<<PVDB_CHANNELS_SHIFT)) ) & PVDB_CHANNELS_MASK )
#define pvdb_root(t)                               (pvdb_levels(t) - 1u)
#define pvdb_log2dim(t, lvl)                       ( ( (t).log2dim_array >> ((lvl)<<PVDB_LOG2DIM_SHIFT) ) & PVDB_LOG2DIM_MASK )
#define pvdb_total_log2dim(t, lvl)                 ( ( (t).total_log2dim_array[((lvl)<<PVDB_TOTAL_LOG2DIM_SHIFT)>>5u] >> ((lvl)<<PVDB_TOTAL_LOG2DIM_SHIFT) ) & PVDB_TOTAL_LOG2DIM_MASK )
#define pvdb_dim(t, lvl)                           (1u << pvdb_log2dim(t, lvl))
#define pvdb_dim_max(t, lvl)                       (pvdb_dim(t, lvl) - 1u)
#define pvdb_dim3(t, lvl)                          (1u << (3u * pvdb_log2dim(t, lvl)))
#define pvdb_dim3_max(t, lvl)                      (pvdb_dim3(t, lvl) - 1u)
#define pvdb_total_dim(t, lvl)                     (1u << pvdb_total_log2dim(t, lvl))
#define pvdb_total_dim3(t, lvl)                    (1u << (3u * pvdb_total_log2dim(t, lvl)))
#define pvdb_total_log2dim_voxel(t, lvl)           ((lvl) == 0u ? 0u : pvdb_total_log2dim(t, (lvl) - uint((lvl) > 0)))
#define pvdb_total_dim_voxel(t, lvl)               (1u << pvdb_total_log2dim_voxel(t, lvl))

/// node math
#define pvdb_is_tile(n)                             (((n) & PVDB_TILE) != 0u)
#define pvdb_mask_word_i(i)                         ((i) >> 5u)
#define pvdb_mask_bit_i(i)                          ((i) & 31u)
#define pvdb_mask_offset(i)                         (PVDB_HEADER_SIZE + pvdb_mask_word_i(i))
#define pvdb_mask_size(t, lvl)                      (1u + ((pvdb_dim3(t, lvl) - 1u) >> 5u))
#define pvdb_data_offset(t, lvl, i)                 (PVDB_HEADER_SIZE + pvdb_mask_size(t, lvl) + (i))
#define pvdb_data_offset_channel(t, lvl, i, ch)     (PVDB_HEADER_SIZE + pvdb_mask_size(t, lvl) + ((ch) * pvdb_dim3(t, lvl)) + (i))
#define pvdb_node_size(t, lvl)                      (pvdb_data_offset(t, lvl, 0) + (pvdb_dim3(t, lvl) * pvdb_channels(t, lvl)))
#define pvdb_leaf_size(t)                           (pvdb_dim3(t, 0) * pvdb_channels(t, 0))


/// NOTE: you must call pvdb_allocate_node(tree, pvdb_root(tree)) after initializing to prepare the tree correctly
///  This call is not included here to avoid requiring access to write functions in order to include this file
PVDB_INLINE void
pvdb_tree_init(
    pvdb_tree_inout         tree,
    uint                    levels,
    PVDB_ARRAY_IN(uint,     log2dim, PVDB_MAX_LEVELS),
    PVDB_ARRAY_IN(uint,     channels, PVDB_MAX_LEVELS))
{
    PVDB_ASSERT(levels <= PVDB_MAX_LEVELS && "too many levels");
    tree.levels_channels_array = levels;
    tree.log2dim_array = 0;
    tree.total_log2dim_array[0] = 0;
    tree.total_log2dim_array[1] = 0;
    uint total = 0;
    for (uint level = 0; level < levels; ++level) {
        tree.log2dim_array |= (log2dim[level] << (level << PVDB_LOG2DIM_SHIFT));
        total += log2dim[level];
        tree.total_log2dim_array[(level << PVDB_TOTAL_LOG2DIM_SHIFT) >> 5u] |= (total << (level << PVDB_TOTAL_LOG2DIM_SHIFT));
        tree.levels_channels_array |= (channels[level] << (PVDB_LEVELS_SHIFT + (level << PVDB_CHANNELS_SHIFT)));
    }
}


/// tree size
PVDB_INLINE uint
pvdb_tree_dense_size(
    pvdb_tree_in                tree)
{
    uint nodes = 1;
    uint size = pvdb_node_size(tree, pvdb_root(tree));
    uint level = pvdb_root(tree) - 1;
    for (;;) {
        nodes *= pvdb_dim3(tree, level);
        size += nodes * pvdb_node_size(tree, level);
        if (level-- == 0) break;
    }
    return size;
}

//endregion

//region coord math

/// local coord <-> index
#define pvdb_coord_local_to_index(t, p, lvl)       pvdb_coord_to_index(p, pvdb_log2dim(t, lvl))
#define pvdb_index_to_coord_local(t, i, lvl)       pvdb_index_to_coord(i, pvdb_log2dim(t, lvl))


/// global coord <-> local coord
#define pvdb_coord_local_to_global(t, p, lvl)      ((p) << int(pvdb_total_log2dim_voxel(t, lvl)))
#define pvdb_coord_global_to_local(t, p, lvl)      (((p) >> int(pvdb_total_log2dim_voxel(t, lvl))) & int(pvdb_dim_max(t, lvl)))


/// global coord -> index
PVDB_INLINE uint
pvdb_coord_global_to_index(
    pvdb_tree_in                tree,
    PVDB_IN(ivec3)              p,
    uint                        level)
{
    const ivec3 local = pvdb_coord_global_to_local(tree, p, level);
    return pvdb_coord_local_to_index(tree, local, level);
}


/// global -> local[]
PVDB_INLINE void
pvdb_coord_global_to_coords(
    pvdb_tree_in           tree,
    PVDB_IN(ivec3)         global,
    PVDB_ARRAY_OUT(ivec3,  local_coords, PVDB_MAX_LEVELS))
{
    for (uint l = 0; l < pvdb_levels(tree); ++l)
        local_coords[l] = pvdb_coord_global_to_local(tree, global, l);
}

//endregion

//region access

#define pvdb_access_xyz(tr, p, lvl)  ((p) & (int(-1) << pvdb_total_log2dim_voxel(tr, lvl)))


struct pvdb_access {
    ivec4 cache[PVDB_MAX_LEVELS-1];
};


PVDB_INLINE void
pvdb_access_clear(
    pvdb_access_out     acc)
{
    for (uint i = 0; i < PVDB_MAX_LEVELS - 1; ++i)
        acc.cache[i].w = 0;
}


PVDB_INLINE bool
pvdb_access_contains(
    pvdb_tree_in        tree,
    pvdb_access_in      acc,
    uint                level,
    PVDB_IN(ivec3)      p)
{
    const ivec4 c = acc.cache[level];
    const ivec3 v = pvdb_access_xyz(tree, p, level);
    return c.w != 0 && c.x == v.x && c.y == v.y && c.z == v.z;
}


PVDB_INLINE uint
pvdb_access_get(
    pvdb_tree_in        tree,
    pvdb_access_in      acc,
    PVDB_IN(ivec3)      p,
    PVDB_OUT(uint)      level)
{
    for (uint l = 0; l < pvdb_levels(tree) - 1; ++l) {
        if (pvdb_access_contains(tree, acc, l, p)){
            level = l;
            return uint(acc.cache[l].w);
        }
    }
    level = pvdb_root(tree);
    return PVDB_ROOT_NODE;
}


PVDB_INLINE void
pvdb_access_set(
    pvdb_tree_in        tree,
    pvdb_access_inout   acc,
    uint                level,
    PVDB_IN(ivec3)      p,
    uint                value)
{
    acc.cache[level] = ivec4(pvdb_access_xyz(tree, p, level), int(value));
    for (uint l = 0; l < level; ++l)
        acc.cache[l].w = 0;
}

//endregion

//region debug nodes

#ifdef PVDB_C

template<uint Log2Dim>
struct pvdb_debug_node {
    static constexpr uint N = 1u << (3 * Log2Dim);
    uint header[PVDB_HEADER_SIZE];
    uint mask[1 + ((N - 1) / 32)];
    uint data[N];
};


static inline void pvdb_debug_nodes()
{
    pvdb_debug_node<1>* n1{};
    pvdb_debug_node<2>* n2{};
    pvdb_debug_node<3>* n3{};
    pvdb_debug_node<4>* n4{};
    pvdb_debug_node<5>* n5{};
    pvdb_debug_node<6>* n6{};
    pvdb_debug_node<7>* n7{};
    pvdb_debug_node<8>* n8{};
    pvdb_debug_node<9>* n9{};
}

#endif

//endregion

#endif //PVDB_TREE_H
