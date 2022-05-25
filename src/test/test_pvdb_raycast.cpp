//
// Created by Djordje on 5/24/2022.
//

#include "catch.hpp"

#define PVDB_C
#define PVDB_ENABLE_PRINTF
#include "../pvdb/pvdb_tree_write.h"
#include "../pvdb/pvdb_raycast.h"

#define PRECISE
//#define CASE1

TEST_CASE("pvdb_raycast", "[pvdb]")
{
    pvdb_debug_nodes();

    pvdb_buf_t<4096> alloc{};
    auto& nodes = (pvdb_buf_t<25600000>&)(*new atom_t[25600000]{});

    pvdb_tree tree{};
    tree.data.nodes = nodes;
    tree.data.alloc = alloc;
    uint l2[PVDB_MAX_LEVELS]{1,2,3};
    uint ch[PVDB_MAX_LEVELS]{1,1,1,1,1,1,1,1};
    pvdb_tree_init(tree, 3, l2, ch);

    pvdb_tree_init_allocator(tree, alloc, 25600000);

#ifdef CASE1
    pvdb_set(tree, {11, 1, 63}, 999u);

#ifdef PRECISE
    pvdb_ray ray{{0.5, 0.5, 0.5}, {0.0, 0.0, 1.0}};
    ivec3 offset = ivec3(11,1,1);
#else
    pvdb_ray ray{{11.5, 1.5, 1.5}, {0.0, 0.0, 1.0}};
    ivec3 offset = ivec3(0,0,0);
#endif

#else
    pvdb_set(tree, {1, 1, 11}, 999u);

#ifdef PRECISE
    pvdb_ray ray{{0.5, 0.5, 0.5}, {0.0, 0.0, -1.0}};
    ivec3 offset = ivec3(1,1,13);
#else
    pvdb_ray ray{{1.5, 1.5, 13.5}, {0.0, 0.0, -1.0}};
    ivec3 offset = ivec3(0,0,0);
#endif

#endif

    pvdb_ray_hit hit;
    if (pvdb_raycast(tree, ray, hit, offset)) {
        printf("HIT: t: %f, voxel: %u, normal: (%f, %f, %f)\n", hit.t, hit.voxel, hit.normal.x, hit.normal.y, hit.normal.z);
    } else {
        printf("MISS\n");
    }
}