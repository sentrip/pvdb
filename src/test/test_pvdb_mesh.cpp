//
// Created by Djordje on 5/25/2022.
//

#include "catch.hpp"

#define PVDB_C
#define PVDB_ENABLE_PRINTF

#include "../pvdb/mesh/pvdb_mesh.h"



struct buddy_alloc {
    uint size{};
    uint used_blocks[32]{};
    uint freelist_count[8]{};
    uint freelist[8][32]{};


    static constexpr uint global_index(uint index, uint local) { return (1u << index) + local - 1u; }

    uint is_used(uint i) const { return (used_blocks[i>>5u] & (1u << (i & 31u))) != 0u; }
    void mark_free2(uint i)    { used_blocks[i>>5u] &= ~(2u << (i & 31u)); }
    void mark_free(uint i)     { used_blocks[i>>5u] &= ~(1u << (i & 31u)); }
    void mark_used(uint i)     { used_blocks[i>>5u] |= (1u << (i & 31u)); }

    uint alloc(uint index) {
        if (freelist_count[index] == 0) {
            const uint ptr = alloc(index + 1);
            const uint buddy = ptr + (1u << index);
            freelist[index][freelist_count[index]++] = ptr;
            freelist[index][freelist_count[index]++] = buddy;
        }
        const uint ptr = freelist[index][--freelist_count[index]];
        mark_used((1u << index) + (ptr >> index) - 1u);
        return ptr;
    }

    void free(uint index, uint ptr) {
        const uint local = ptr >> index;
        const uint buddy_local = local + (~local & 1u) - (local & 1u);
        const uint buddy_global = global_index(index, buddy_local);
        if (!is_used(buddy_global)) {
            const uint ptr_buddy = buddy_local << index;
            freelist[index + 1][freelist_count[index + 1]++] = min(ptr, ptr_buddy);
            mark_free2(min(global_index(index, local), buddy_global));
        } else {
            freelist[index][freelist_count[index]++] = ptr;
            mark_free(global_index(index, local));
        }
    }
};


template<uint Levels>
struct buddy_alloc2 {
    uint    size{};
    atom_t  freelist_begin[Levels]{};
    atom_t  freelist_end[Levels]{};
    atom_t* used_blocks{};
    atom_t* freelist[Levels]{};

    explicit buddy_alloc2(uint size = 512) : size{size} {
        used_blocks = new atom_t[size >> 5u]{};
        for (uint l = 0; l < Levels; ++l) {
            freelist[l] = new atom_t[size >> l]{};
        }
        freelist_insert(Levels-1, 0u);
    }

    ~buddy_alloc2() {
        delete[] used_blocks;
        for (uint l = 0; l < Levels; ++l)
            delete[] freelist[l];
    }

    uint alloc(uint level) {
        uint ptr = 0;
        if (!freelist_try_pop(level, ptr)) {
            ptr = alloc(level + 1u);
            freelist_insert2(level, ptr, ptr + (1u << level));
            while (!freelist_try_pop(level, ptr)) {}
        }
        mark_used(mask_index(level, ptr_to_index(level, ptr)));
        return ptr;
    }

    uint alloc_stack(uint level) {
        uint ptr = 0;
        for (;;) {
            if (freelist_try_pop(level, ptr))
                break;
            for (uint l = Levels - 1; l > level; --l) {
                if (freelist_try_pop(l, ptr))
                    freelist_insert2(l-1u, ptr, ptr + (1u << (l-1u)));
            }
            if (freelist_try_pop(level, ptr))
                break;
        }
        mark_used(mask_index(level, ptr_to_index(level, ptr)));
        return ptr;
    }

    void free(uint level, uint ptr) {
        const uint i = ptr_to_index(level, ptr);
        if (!is_used(mask_index(level, buddy_index(i)))) {
            const uint parent_index = i & ~1u;
            freelist_insert(level + 1u, index_to_ptr(level, parent_index));
            mark_free(mask_index(level, parent_index));
        } else {
            freelist_insert(level, ptr);
            mark_free(mask_index(level, i));
        }
    }

    static constexpr uint index_to_ptr(uint level, uint i)  { return i << level; }
    static constexpr uint ptr_to_index(uint level, uint ptr){ return ptr >> level; }
    static constexpr uint buddy_index(uint i)               { return i + (~i & 1u) - (i & 1u); }
    static constexpr uint mask_index(uint level, uint i)    { return ((1u << level) + i - 1u) >> 1u; }

    bool is_used(uint i) const { return   (used_blocks[i>>5u] & (1u << (i & 31u))) != 0u; }
    void mark_free(uint i)     { atomicAnd(used_blocks[i>>5u], ~(1u << (i & 31u))); }
    void mark_used(uint i)     { atomicOr( used_blocks[i>>5u],  (1u << (i & 31u))); }

    void freelist_insert(uint level, uint ptr) {
        for (;;) {
            const uint end = freelist_end[level];
            const uint new_end = (end + 1u) & ((size >> level) - 1u);
            if (atomicCompSwap(freelist_end[level], end, new_end) == end) {
                freelist[level][end] = ptr;
                return;
            }
        }
    }

    void freelist_insert2(uint level, uint ptr1, uint ptr2) {
        for (;;) {
            const uint end = freelist_end[level];
            const uint max_i = (size >> level) - 1u;
            const uint safe_end = end == max_i ? 0u : end;
            const uint new_end = (safe_end + 2u) & max_i;
            if (atomicCompSwap(freelist_end[level], end, new_end) == end) {
                freelist[level][safe_end] = ptr1;
                freelist[level][safe_end + 1u] = ptr2;
                return;
            }
        }
    }

    bool freelist_try_pop(uint level, uint& ptr) {
        for (;;) {
            const uint begin = freelist_begin[level];
            const uint end = freelist_end[level];
            if (begin == end) return false;
            const uint value = freelist[level][begin];
            const uint new_begin = (begin + 1u) & ((size >> level) - 1u);
            if (atomicCompSwap(freelist_begin[level], begin, new_begin) == begin) {
                ptr = value;
                return true;
            }
        }
        return false;
    }
};

//region definitions

#define PVDB_BA_MAX_LEVELS      32u

#define pvdb_buddy_alloc_at(buf, addr)  pvdb_buf_at(buf, PVDB_GLOBAL_BUDDY_ALLOC, addr)

PVDB_INLINE uint
pvdb_buddy_alloc_init(
    pvdb_buf_inout      alloc,
    uint                size,
    uint                levels);


PVDB_INLINE uint
pvdb_buddy_alloc_alloc(
    pvdb_buf_inout      alloc,
    uint                level);


PVDB_INLINE void
pvdb_buddy_alloc_free(
    pvdb_buf_inout      alloc,
    uint                level,
    uint                ptr);


//endregion

//region implementation

#define PVDB_BA_MEMBER_SIZE                 0u
#define PVDB_BA_MEMBER_LEVELS               1u
#define PVDB_BA_MEMBER_FREELIST_BEGIN       2u
#define PVDB_BA_MEMBER_FREELIST_END         (PVDB_BA_MEMBER_FREELIST_BEGIN + PVDB_BA_MAX_LEVELS)
#define PVDB_BA_MEMBER_USED_BLOCKS          (PVDB_BA_MEMBER_FREELIST_END + PVDB_BA_MAX_LEVELS)

#define pvdb_ba_index_to_ptr(level, i)      ((i) << (level))
#define pvdb_ba_ptr_to_index(level, ptr)    ((ptr) >> (level))
#define pvdb_ba_buddy_index(i)              ((i) + (~(i) & 1u) - ((i) & 1u))
#define pvdb_ba_mask_index(level, i)        ((1u << (level)) + (i) - 1u)


//uint    size{};
//atom_t  freelist_begin[Levels]{};
//atom_t  freelist_end[Levels]{};
//atom_t* used_blocks{};
//atom_t* freelist[Levels]{};

PVDB_INLINE bool pvdb_buddy_alloc_read_mask(pvdb_buf_in alloc, uint begin, uint i) {
    return (alloc[begin+(i>>5u)] & (1u << (i & 31u))) != 0u;
}

PVDB_INLINE bool pvdb_buddy_alloc_write_mask(pvdb_buf_inout alloc, uint begin, uint i, uint m) {
    const uint address = begin + (i >> 5u);
    const uint mask = m << (i & 31u);
    const uint current = pvdb_buddy_alloc_at(alloc, address);
    const uint updated = (m != 0u) ? (current | mask) : (current & ~mask);
    const uint prev = atomicCompSwap(pvdb_buddy_alloc_at(alloc, address), current, updated);
    return prev == current && (current != updated);
}

PVDB_INLINE bool pvdb_buddy_alloc_is_used(pvdb_buf_in alloc, uint index) {
    return pvdb_buddy_alloc_read_mask(alloc, PVDB_BA_MEMBER_USED_BLOCKS, index);
}

PVDB_INLINE void pvdb_buddy_alloc_mark_used(pvdb_buf_inout alloc, uint index) {
    while (!pvdb_buddy_alloc_write_mask(alloc, PVDB_BA_MEMBER_USED_BLOCKS, index, 1u))
        ;
}

PVDB_INLINE void pvdb_buddy_alloc_mark_used2(pvdb_buf_inout alloc, uint index) {
    while (!pvdb_buddy_alloc_write_mask(alloc, PVDB_BA_MEMBER_USED_BLOCKS, index, 2u))
        ;
}

PVDB_INLINE bool pvdb_buddy_alloc_freelist_empty(pvdb_buf_in alloc, uint level) {
    return false;
}

PVDB_INLINE void pvdb_buddy_alloc_freelist_insert(pvdb_buf_inout alloc, uint level, uint ptr) {

}

PVDB_INLINE void pvdb_buddy_alloc_freelist_insert2(pvdb_buf_inout alloc, uint level, uint ptr1, uint ptr2) {

}

PVDB_INLINE bool pvdb_buddy_alloc_freelist_try_pop(pvdb_buf_inout alloc, uint level, PVDB_INOUT(uint) ptr) {
    return false;
}


uint
pvdb_buddy_alloc_init(
    pvdb_buf_inout      alloc,
    uint                size,
    uint                levels)
{
    pvdb_buddy_alloc_at(alloc, PVDB_BA_MEMBER_SIZE) = size;
    pvdb_buddy_alloc_at(alloc, PVDB_BA_MEMBER_LEVELS) = levels;
    uint total_size = 2u + 2u * PVDB_BA_MAX_LEVELS + (1u + ((size - 1u) / 32u));
    for (uint l = 0; l < levels; ++l)
        total_size += size >> l;
    return total_size;
}


uint
pvdb_buddy_alloc_alloc(
    pvdb_buf_inout      alloc,
    uint                level)
{
    uint ptr = 0;
    const uint levels = pvdb_buddy_alloc_at(alloc, PVDB_BA_MEMBER_LEVELS);
    uint current_level = levels - 1u;
    for (;;) {
        if (!pvdb_buddy_alloc_freelist_empty(alloc, level) && pvdb_buddy_alloc_freelist_try_pop(alloc, level, ptr))
            break;

        for (uint l = current_level; l > level; --l) {
            if (!pvdb_buddy_alloc_freelist_empty(alloc, l) && pvdb_buddy_alloc_freelist_try_pop(alloc, l, ptr)) {
                pvdb_buddy_alloc_mark_used2(alloc, pvdb_ba_mask_index(l, pvdb_ba_ptr_to_index(l, ptr)));
                pvdb_buddy_alloc_freelist_insert2(alloc, l-1u, ptr, ptr + (1u << (l-1u)));
                current_level = l;
            }
        }
    }
    pvdb_buddy_alloc_mark_used(alloc, pvdb_ba_mask_index(level, pvdb_ba_ptr_to_index(level, ptr)));
    return ptr;
}


void
pvdb_buddy_alloc_free(
    pvdb_buf_inout      alloc,
    uint                level,
    uint                ptr)
{
    uint i = pvdb_ba_ptr_to_index(level, ptr);
    while (!pvdb_buddy_alloc_is_used(alloc, pvdb_ba_mask_index(level, pvdb_ba_buddy_index(i)))) {
        const uint parent_index = i & ~1u;
        if (!pvdb_buddy_alloc_write_mask(alloc, PVDB_BA_MEMBER_USED_BLOCKS, pvdb_ba_mask_index(level, parent_index), 0u))
            continue;
        ptr = pvdb_ba_index_to_ptr(level, parent_index);
        i = parent_index;
        ++level;
    }
    pvdb_buddy_alloc_freelist_insert(alloc, level, ptr);

//    const uint i = pvdb_ba_ptr_to_index(level, ptr);
//    if (!pvdb_buddy_alloc_is_used(alloc, pvdb_ba_mask_index(level, pvdb_ba_buddy_index(i)))) {
//        const uint parent_index = i & ~1u;
//        pvdb_buddy_alloc_freelist_insert(alloc, level + 1u, pvdb_ba_index_to_ptr(level, parent_index));
//        pvdb_buddy_alloc_mark_free(alloc, pvdb_ba_mask_index(level, parent_index));
//    } else {
//        pvdb_buddy_alloc_freelist_insert(alloc, level, ptr);
//        pvdb_buddy_alloc_mark_free(alloc, pvdb_ba_mask_index(level, i));
//    }
}

//endregion


TEST_CASE("pvdb_mesh", "[pvdb]")
{
//    buddy_alloc alloc{512};
//    alloc.freelist[7][alloc.freelist_count[7]++] = 0u;

    buddy_alloc2<4> alloc{32};

    printf("%u\n", alloc.alloc_stack(0));
    printf("%u\n", alloc.alloc_stack(0));
    alloc.free(0, 1);
    printf("%u\n", alloc.alloc_stack(0));
    printf("%u\n", alloc.alloc_stack(0));
    printf("%u\n", alloc.alloc_stack(0));


//    pvdb_buf_t<256> meshes{};
//    pvdb_buf_t<1024> alloc{};
//    auto& vertices = (pvdb_buf_t<128000>&)*new atom_t[1000000];
//
//    pvdb_allocator_level levels[PVDB_ALLOCATOR_MAX_LEVELS]{};
//    for (uint i = 0; i < 16; ++i) levels[i] = {1u << (i+PVDB_MESH_INITIAL_CAPACITY_LOG2DIM), 1u, 0u};
//    uint alloc_buf_size = 0;
//    pvdb_allocator_init(alloc, 16, levels, 0u, alloc_buf_size);
//
//    uint dim3 = 1u << (3u * PVDB_MESH_LOG2DIM);
//    for (uint i = 0; i < dim3; ++i)
//        for (uint d = 0; d < 6; ++d)
//            pvdb_mesh_add_face(meshes, alloc, vertices, 0u, pvdb_vertex_make(1u, i, d, 0u, 0u));
}