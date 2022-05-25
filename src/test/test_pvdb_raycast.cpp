//
// Created by Djordje on 5/24/2022.
//

#include "catch.hpp"

#define PVDB_C
#define PVDB_ENABLE_PRINTF
#include "../pvdb/pvdb_tree_write.h"
#include "../pvdb/pvdb_raycast.h"


TEST_CASE("pvdb_raycast", "[pvdb]")
{
    pvdb_debug_nodes();

    pvdb_buf_t<256> alloc{};
    auto& nodes = (pvdb_buf_t<1000000>&)(*new atom_t[1000000]{});
    auto& atlas = (pvdb_buf_t<25600000>&)(*new atom_t[25600000]{});

    pvdb_tree tree{};
    tree.data.nodes = nodes;
    tree.data.alloc = alloc;
    tree.data.atlas = atlas;
    uint l2[PVDB_MAX_LEVELS]{1,2,3};
    uint ch[PVDB_MAX_LEVELS]{1,1,1,1,1,1,1,1};
    #ifdef PVDB_USE_IMAGES
    pvdb_tree_init(tree, 3, l2, 8u);
    #else
    pvdb_tree_init(tree, 3, l2, ch);
    #endif

    pvdb_tree_init_allocator(tree, alloc, 256, 8u);

    for (int x = 0; x <= 15; ++x)
        for (int z = 0; z <= 15; ++z)
            pvdb_set(tree, {x, 0, z}, 999u);

    pvdb_ray ray{{0.73, 2.05, 0.46}, {-0.0008, 0.2729, 0.9620}};
    pvdb_ray_hit hit;
    if (pvdb_raycast(tree, ray, hit, ivec3(0,0,0))) {
        printf("HIT: t: %f, voxel: %u, normal: (%f, %f, %f)\n", hit.t, hit.voxel, hit.normal.x, hit.normal.y, hit.normal.z);
    } else {
        printf("MISS\n");
    }
}