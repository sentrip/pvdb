#version 450

#define LOCAL_SIZE  8
#define PVDB_GENERATE_CHANNEL_NODE 1u

layout (local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = LOCAL_SIZE) in;

#include "pvdb_math.h"
#include "pvdb_tree_write.h"

uint pvdb_generate_voxel(in ivec3 p)
{
    if (p.y > 0) return 0u;
    return 1u + uint(p.x & 1) + uint(p.z & 1);
}

void pvdb_tree_did_generate(in pvdb_tree tree, uint level, uint node, ivec3 p)
{

}

void pvdb_tree_enqueue_erase(in pvdb_tree tree, uint level, uint parent, uint index, uint child)
{

}


layout(push_constant) uniform constants {
    ivec3 cur;
    ivec3 prev;
    ivec3 offset;
    uint  level;
    pvdb_tree tree;
} PC;

void main() {
    ivec3 i = ivec3(gl_WorkGroupID);
    ivec3 size = ivec3(gl_NumWorkGroups);
    ivec3 xyz_load = ivec3(0), xyz_unload = ivec3(0);
    const ivec3 local = ivec3(gl_LocalInvocationID) + (ivec3(PC.offset) * LOCAL_SIZE);

    if (pvdb_region_test(i, size, PC.cur, PC.prev, pvdb_total_log2dim(PC.tree, PC.level), xyz_load)) {
        const ivec3 p = xyz_load + local;
        const uint v = pvdb_generate_voxel(p);
        const uint node = pvdb_insert(PC.tree, PC.level, xyz_load, 0u);

        if (PC.level == 0u) {
            pvdb_write_leaf(PC.tree, node, local, v);
        } else {
            pvdb_write_node(PC.tree, PC.level, node, pvdb_coord_global_to_index(PC.tree, p), v, PVDB_GENERATE_CHANNEL_NODE);
        }

        if (gl_LocalInvocationIndex == 0 && offset.x == 0 && offset.y == 0 && offset.z == 0) {
            pvdb_tree_did_generate(PC.tree, PC.level, node, xyz_load);
        }
    }

    if (PC.prev.x != PVDB_INT_MAX && pvdb_region_test(i, size, PC.prev, PC.cur, pvdb_total_log2dim(PC.tree, PC.level), xyz_unload)) {
        const uint node = pvdb_traverse_at_least(PC.tree, xyz_unload, PC.level);
        if (node != PVDB_ROOT_NODE) {
            // TODO: enqueue save then execute erase after save explicitly with vulkan
            const uint parent = pvdb_get_parent(PC.tree, node);
            const uint index = pvdb_coord_local_to_index(PC.tree, local);
            pvdb_tree_enqueue_erase(PC.tree, PC.level, parent, index, node);
        }
    }
}
