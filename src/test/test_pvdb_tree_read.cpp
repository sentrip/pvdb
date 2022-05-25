//
// Created by Djordje on 5/24/2022.
//

#include "catch.hpp"

#define PVDB_C
#include "../pvdb/pvdb_tree_read.h"

template<uint NWords, uint NLevels>
pvdb_tree make_test_tree_read(pvdb_buf_t<NWords>& buf, PVDB_ARRAY_IN(uint, log2dim, NLevels))
{
    pvdb_tree tree{};
    tree.data.nodes = buf;
    uint l2[PVDB_MAX_LEVELS]{};
    uint ch[PVDB_MAX_LEVELS]{1,1,1,1,1,1,1,1};
    for (uint i = 0; i < NLevels; ++i) l2[i] = log2dim[i];
    pvdb_tree_init(tree, NLevels, l2, ch);
    return tree;
}


template<uint NLevels>
void test_pvdb_coord_math(PVDB_ARRAY_IN(uint, log2dim, NLevels))
{
    pvdb_buf_t<8> data{};
    auto tree = make_test_tree_read(data, log2dim);

    // Get max coord and coord that is 1,1,1 in local space at every level
    ivec3 max{1,1,1}, all_one{};
    for (uint level = 0; level < NLevels; ++level) {
        all_one += pvdb_coord_local_to_global(tree, ivec3(1,1,1), level);
        max *= ivec3(int(pvdb_dim(tree, level)));
    }
    max += ivec3(-1);

    // Check all_one and max coord for every level
    uint level = NLevels - 1;
    for (;;) {
        auto one = pvdb_coord_global_to_local(tree, all_one, level);
        REQUIRE( 1 == one.x );
        REQUIRE( 1 == one.y );
        REQUIRE( 1 == one.z );
        auto mx = pvdb_coord_global_to_local(tree, max, level);
        auto m = pvdb_dim_max(tree, level);
        REQUIRE( m == mx.x );
        REQUIRE( m == mx.y );
        REQUIRE( m == mx.z );
        if (level-- == 0) break;
    }
}


template<uint NLevels>
void test_pvdb_traverse(PVDB_ARRAY_IN(uint, log2dim, NLevels))
{
    constexpr uint VALUE = 999u;

    pvdb_buf_t<64000> data{};
    auto tree = make_test_tree_read(data, log2dim);

    // Beginning of tree reserved for root node
    uint offset = pvdb_node_size(tree, pvdb_root(tree));

    // Get expected nodes for each level
    uint nodes[NLevels]{};
    nodes[pvdb_root(tree)] = PVDB_ROOT_NODE;
    uint level = pvdb_root(tree) - 1;
    for (;;) {
        nodes[level] = offset;
        offset += pvdb_node_size(tree, level);
        if (level-- == 0) break;
    }

    // Set nodes and masks in tree
    const uint mask_offset = pvdb_mask_offset(0);
    for (uint i = 0; i < NLevels; ++i) {
        const uint data_offset = pvdb_data_offset(tree, i, 0);
        data[nodes[i] + mask_offset] = 0x1;
        data[nodes[i] + data_offset] = i == 0 ? VALUE : nodes[i - 1];
    }

    // For every level, traverse down to every level below
    uint start_level = pvdb_root(tree);
    for (;;) {
        uint end_level = start_level - 1;
        for (;;) {
            uint target_level = end_level;
            uint result = pvdb_traverse(tree, nodes[start_level], start_level, target_level, {});
            REQUIRE( result == nodes[end_level] );
            REQUIRE( target_level == end_level );
            if (end_level-- == 0) break;
        }
        if (start_level-- == 1) break;
    }
}


TEST_CASE("pvdb_coord_math", "[pvdb]")
{
    test_pvdb_coord_math({2, 2});
    test_pvdb_coord_math({2, 2, 2, 2});
    test_pvdb_coord_math({4, 4, 4, 4, 4, 4});
    test_pvdb_coord_math({2, 3});
    test_pvdb_coord_math({2, 3, 4, 5});
    test_pvdb_coord_math({4, 5, 6, 7, 8, 9});
}

TEST_CASE("pvdb_traverse", "[pvdb]")
{
    test_pvdb_traverse({2, 2});
    test_pvdb_traverse({2, 2, 2, 2});
    test_pvdb_traverse({4, 4, 4, 4, 4, 4});
    test_pvdb_traverse({2, 3});
    test_pvdb_traverse({2, 3, 4, 5});
}
