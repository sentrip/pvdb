//
// Created by Djordje on 5/24/2022.
//

#include "catch.hpp"

#define PVDB_C
#include "../pvdb/pvdb_tree_write.h"


template<uint NWords, uint NWordsAtlas, uint NLevels>
pvdb_tree make_test_tree_write(pvdb_buf_t<NWords>& buf, pvdb_buf_t<256>& buf_alloc, pvdb_buf_t<NWordsAtlas>& buf_atlas, PVDB_ARRAY_IN(uint, log2dim, NLevels), uint leaf_count_log2dim)
{
    pvdb_tree tree{};
    tree.data.nodes = buf;
    tree.data.alloc = buf_alloc;
    tree.data.atlas = buf_atlas;
    uint l2[PVDB_MAX_LEVELS]{};
    uint ch[PVDB_MAX_LEVELS]{1,1,1,1,1,1,1,1};
    for (uint i = 0; i < NLevels; ++i) l2[i] = log2dim[i];
    #ifdef PVDB_USE_IMAGES
    pvdb_tree_init(tree, NLevels, l2, leaf_count_log2dim);
    #else
    pvdb_tree_init(tree, NLevels, l2, ch);
    #endif

    pvdb_tree_init_allocator(tree, buf_alloc, NWords, leaf_count_log2dim);

    return tree;
}


template<uint NLevels>
void test_pvdb_insert_node(PVDB_ARRAY_IN(uint, log2dim, NLevels), const ivec3& p)
{
    pvdb_buf_t<256u> alloc{};
    pvdb_buf_t<64000> data{}, atlas{};
    pvdb_tree tree = make_test_tree_write(data, alloc, atlas, log2dim, 3u);

    // Beginning of tree reserved for root node
    uint offset = pvdb_node_size(tree, pvdb_root(tree));

    // Get expected nodes for each level
    uint nodes[NLevels]{};
    nodes[pvdb_root(tree)] = PVDB_ROOT_NODE;
    uint level = pvdb_root(tree) - 1;
    for (;;) {
        #ifdef PVDB_USE_IMAGES
        nodes[level] = level == 0u ? 1u : offset;
        #else
        nodes[level] = offset;
        #endif
        if (level > 0u)
            offset += pvdb_node_size(tree, level);
        if (level-- == 0) break;
    }

    // for every level, insert and check that full traverse goes down to this level
    level = pvdb_root(tree);
    for (;;) {
        // Traverse fails at previous level
        uint target_level = 0;
        uint result = pvdb_traverse(tree, 0, pvdb_root(tree), target_level, p);
        REQUIRE( target_level == level );
        REQUIRE( result == (level == pvdb_root(tree) ? 0u : nodes[level]) );

        // Insert node
        auto l = pvdb_coord_global_to_index(tree, p, level);
        uint n = pvdb_try_insert_node(tree, level - 1, nodes[level], l, 0);
        REQUIRE( n == nodes[level - 1] );

        // Traverse succeeds at current level
        target_level = 0;
        result = pvdb_traverse(tree, 0, pvdb_root(tree), target_level, p);
        REQUIRE( target_level == (level - 1) );
        REQUIRE( result == nodes[level - 1] );

        if (level-- == 1) break;
    }
}

template<uint NLevels>
void test_pvdb_insert(PVDB_ARRAY_IN(uint, log2dim, NLevels))
{
    pvdb_buf_t<256u> alloc{};
    pvdb_buf_t<64000> data{}, atlas{};
    pvdb_tree tree = make_test_tree_write(data, alloc, atlas, log2dim, 16u);

    ivec4 results[NLevels]{};

    // zero
    auto p = ivec3(0);
    uint n = pvdb_insert(tree, 0, p, 0);
    uint rs = pvdb_traverse_at_least(tree, 0, p);
    REQUIRE( n == rs );
    results[0] = ivec4(p, int(rs));

    // local + 1
    for (uint level = 0; level < NLevels - 1; ++level) {
        p = ivec3(0, 0, pvdb_total_dim(tree, level));
        uint node = pvdb_insert(tree, 0, p, 0);
        uint result = pvdb_traverse_at_least(tree, 0, p);
        REQUIRE( node == result );
        results[level + 1] = ivec4(p, int(result));
    }

    // ensure inserts do not affect other inserts
    auto r = results[0];
    uint node = pvdb_traverse_at_least(tree, 0, {r.x, r.y, r.z});
    REQUIRE( node == uint(r.w) );
    for (uint level = 0; level < NLevels - 1; ++level) {
        r = results[level + 1];
        node = pvdb_traverse_at_least(tree, 0, {r.x, r.y, r.z});
        REQUIRE( node == uint(r.w) );
    }

    // ensure repeated inserts do nothing
    uint allocated = tree.data.alloc[0];
    r = results[0];
    node = pvdb_insert(tree, 0, {r.x, r.y, r.z}, 0);
    REQUIRE( node == uint(r.w) );
    for (uint level = 0; level < NLevels - 1; ++level) {
        r = results[level + 1];
        node = pvdb_insert(tree, 0, {r.x, r.y, r.z}, 0);
        REQUIRE( node == uint(r.w) );
    }
    REQUIRE(tree.data.alloc[0] == allocated);
}

template<uint NLevels>
void test_pvdb_set(PVDB_ARRAY_IN(uint, log2dim, NLevels), uint delta, uint steps)
{
    pvdb_buf_t<256u> alloc{};
#ifdef PVDB_USE_IMAGES
    auto& data = (pvdb_buf_t<10000000>&)(*new atom_t[10000000]{});
    auto& atlas = (pvdb_buf_t<20000000>&)(*new atom_t[20000000]{});
#else
    auto& data = (pvdb_buf_t<10000000>&)(*new atom_t[10000000]{});
    auto& atlas = (pvdb_buf_t<200>&)(*new atom_t[200]{});
#endif
    pvdb_tree tree = make_test_tree_write(data, alloc, atlas, log2dim, 16u);

    static constexpr auto get_val = [](int x, int y, int z) {
        return uint(x * 1000000 + y * 1000 + z);
    };

    const int end = std::min(delta * steps, pvdb_total_dim(tree, pvdb_root(tree)));
    for (int x = 0; x < end; x += int(delta)) {
        for (int y = 0; y < end; y += int(delta)) {
            for (int z = 0; z < end; z += int(delta)) {
                pvdb_set(tree, {x, y, z}, get_val(x, y, z));
                REQUIRE( get_val(x, y, z) == pvdb_get(tree, {x, y, z}) );
            }
        }
    }
    const int mx = int(pvdb_total_dim(tree, pvdb_root(tree)) - 1);
    if (mx != int(steps*delta - 1))
        pvdb_set(tree, {mx, mx, mx}, 99999999);

    for (int x = 0; x < end; x += int(delta)) {
        for (int y = 0; y < end; y += int(delta)) {
            for (int z = 0; z < end; z += int(delta)) {
                REQUIRE( get_val(x, y, z) == pvdb_get(tree, {x, y, z}) );
            }
        }
    }

    if (mx != int(steps*delta - 1))
        REQUIRE( 99999999 == pvdb_get(tree, {mx, mx, mx}) );

    delete [] ((atom_t*)&data);
    delete [] ((atom_t*)&atlas);
}


TEST_CASE("pvdb_insert_node", "[pvdb]")
{
    test_pvdb_insert_node({2, 2}            , {0});
    test_pvdb_insert_node({2, 2, 2, 2}      , {0});
    test_pvdb_insert_node({4, 4, 4, 4, 4, 4}, {0});
    test_pvdb_insert_node({2, 3}            , {0});
    test_pvdb_insert_node({2, 3, 4, 5}      , {0});
}

TEST_CASE("pvdb_insert", "[pvdb]")
{
    test_pvdb_insert({2, 2});
    test_pvdb_insert({2, 2, 2, 2});
    test_pvdb_insert({2, 3});
    test_pvdb_insert({2, 3, 4, 5});
}

TEST_CASE("pvdb_set", "[pvdb]")
{
    test_pvdb_set({2, 2}, 1, 8);
    test_pvdb_set({2, 2}, 4, 4);
    test_pvdb_set({2, 2, 2}, 16, 4);
    test_pvdb_set({2, 2, 2, 2}, 32, 8);
    test_pvdb_set({2, 3, 4}, 6, 4);
    test_pvdb_set({1, 1, 1, 1}, 1, 16);
}
