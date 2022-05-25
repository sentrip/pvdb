/** MACRO INPUTS
    * PVDB_TREE_READONLY    - whether the tree will only be read or not
*/

#define PVDB_GLOBAL_TREE            PVDBTree
#define PVDB_GLOBAL_ALLOC           PVDBAlloc
#define PVDB_GLOBAL_TREE_ATLAS      PVDBTreeAtlas

#include "pvdb_tree.h"

//////////////////////////////////////////////////////////
////  READ  //////////////////////////////////////////////
//////////////////////////////////////////////////////////
#ifdef PVDB_TREE_READONLY
#define PVDB_TQ readonly
#else
#define PVDB_TQ coherent
#endif


layout(std430, binding = PVDB_BINDING_TREE, set = 0)
    PVDB_TQ  restrict buffer PVDBTreeT { uint data[]; } PVDB_GLOBAL_TREE[PVDB_MAX_TREES];

#ifdef PVDB_USE_IMAGES

#if   (PVDB_CHANNELS_LEAF == 1)
#define PVDB_ATLAS_FORMAT           r32ui
#elif (PVDB_CHANNELS_LEAF == 2)
#define PVDB_ATLAS_FORMAT           rg32ui
#elif (PVDB_CHANNELS_LEAF == 3)
#define PVDB_ATLAS_FORMAT           rgb32ui
#elif (PVDB_CHANNELS_LEAF == 4)
#define PVDB_ATLAS_FORMAT           rgba32ui
#else
#error PVDB_CHANNELS_LEAF can only be (1, 2, 3, 4)
#endif

layout(PVDB_ATLAS_FORMAT, binding = PVDB_BINDING_TREE_ATLAS, set = 0)
    PVDB_TQ  restrict uniform uimage3D PVDB_GLOBAL_TREE_ATLAS[PVDB_MAX_TREES];

#endif

#include "pvdb_tree_read.h"


//////////////////////////////////////////////////////////
////  WRITE  /////////////////////////////////////////////
//////////////////////////////////////////////////////////
#ifndef PVDB_TREE_READONLY

layout(std430, binding = PVDB_BINDING_TREE_ALLOC, set = 0)
    coherent restrict buffer PVDBAllocT { uint data[]; } PVDB_GLOBAL_ALLOC[PVDB_MAX_TREES];

#define PVDB_ALLOCATOR_MASK
#include "pvdb_tree_write.h"

#endif
