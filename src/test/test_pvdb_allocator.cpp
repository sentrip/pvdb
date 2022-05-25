//
// Created by Djordje on 5/25/2022.
//

#include "catch.hpp"

#define PVDB_C
#define PVDB_ALLOCATOR_MASK
#include "../pvdb/pvdb_allocator.h"

TEST_CASE("allocator", "[pvdb]")
{
    pvdb_buf_t<16000> alloc{};
    pvdb_allocator_init(alloc, 1, {{1u, 64u, 0u}}, 0u);

    uint ptrs[64]{};
    for (uint i = 0; i < 64; ++i)
        ptrs[i] = pvdb_allocator_alloc(alloc, 0);

    for (uint i = 0; i < 64; ++i)
        REQUIRE(ptrs[i] == i);

    for (uint i = 0; i < 64; i += 2)
        pvdb_allocator_free(alloc, 0, ptrs[i]);

    for (uint i = 0; i < 64; i += 2)
        REQUIRE(pvdb_allocator_alloc(alloc, 0) == ptrs[i]);
}
