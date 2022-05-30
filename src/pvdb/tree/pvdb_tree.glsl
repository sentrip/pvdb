/** MACRO INPUTS
    * PVDB_TREE_READONLY    - whether the tree will only be read or not
*/

#define PVDB_GLOBAL_TREE            PVDBTree
#define PVDB_GLOBAL_ALLOC           PVDBAlloc

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
