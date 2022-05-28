//
// Created by Djordje on 5/12/2022.
//
/** MACRO INPUTS
    * PVDB_GLOBAL_ALLOC     - name of global array (only required in glsl)
    * PVDB_ALLOCATOR_LINEAR - use linear allocator algorithm
    * PVDB_ALLOCATOR_MASK   - use mask allocator algorithm
*/

#ifndef PVDB_ALLOCATOR_H
#define PVDB_ALLOCATOR_H

#include "pvdb_buffer.h"

//region definitions

#ifndef PVDB_GLOBAL_ALLOC
#define PVDB_GLOBAL_ALLOC                   GlobalAlloc
#endif

#define PVDB_ALLOCATOR_MAX_LEVELS           16u

#define pvdb_alloc_at(a, addr)              pvdb_buf_at(a, PVDB_GLOBAL_ALLOC, addr)

//endregion

//region init

struct pvdb_allocator_level {
    uint            block_size;
    uint            max_allocations;
    uint            data_offset;
};


PVDB_INLINE void
pvdb_allocator_init_level(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                n_levels,
    uint                max_allocations,
    uint                block_size,
    uint                data_offset,
    PVDB_INOUT(uint)    offset,
    PVDB_INOUT(uint)    implementation_offset);


PVDB_INLINE uint
pvdb_allocator_init(
    pvdb_buf_inout                      alloc,
    uint                                n_levels,
    PVDB_ARRAY_IN(pvdb_allocator_level, levels, PVDB_ALLOCATOR_MAX_LEVELS),
    uint                                initial_offset,
    PVDB_INOUT(uint)                    implementation_size)
{
    PVDB_ASSERT(n_levels <= PVDB_ALLOCATOR_MAX_LEVELS && "too many levels");
    uint size = 0;
    uint offset = initial_offset;
    for (uint l = 0; l < n_levels; ++l) {
        const uint specific_offset = (levels[l].data_offset == -1u) ? offset : levels[l].data_offset;
        pvdb_allocator_init_level(alloc, l, n_levels, levels[l].max_allocations, levels[l].block_size, specific_offset, size, implementation_size);
        if (levels[l].data_offset == -1u)
            offset += levels[l].max_allocations * levels[l].block_size;
    }
    return size;
}

//endregion

#ifdef PVDB_ALLOCATOR_LINEAR

//region linear

struct pvdb_allocator {
    uint            count;
    uint            max_allocations;
    uint            block_size;
    uint            data_offset;
};


void
pvdb_allocator_init_level(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                n_levels,
    uint                max_allocations,
    uint                block_size,
    uint                data_offset,
    PVDB_INOUT(uint)    offset,
    PVDB_INOUT(uint)    implementation_offset)
{
    const uint begin = (index * 4);
    pvdb_alloc_at(alloc, begin + 0) = 0;
    pvdb_alloc_at(alloc, begin + 1) = max_allocations;
    pvdb_alloc_at(alloc, begin + 2) = block_size;
    pvdb_alloc_at(alloc, begin + 3) = data_offset;
    offset += 4;
    implementation_offset += 4;
}


PVDB_INLINE uint
pvdb_allocator_alloc(
    pvdb_buf_inout      alloc,
    uint                index)
{
    const uint n                = atomicAdd(pvdb_alloc_at(alloc, index * 4), 1);
    const uint max_allocations  = pvdb_alloc_at(alloc, (index * 4) + 1);
    const uint block_size       = pvdb_alloc_at(alloc, (index * 4) + 2);
    const uint data_offset      = pvdb_alloc_at(alloc, (index * 4) + 3);
    return data_offset + ((n % max_allocations) * block_size);
}


PVDB_INLINE void
pvdb_allocator_free(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                ptr)
{}

//endregion

#elif defined(PVDB_ALLOCATOR_MASK)

//region mask

struct pvdb_allocator {
    uint            first_free_word;
    uint            max_allocations;
    uint            block_size;
    uint            data_offset;
    uint            mask_offset;
};


void
pvdb_allocator_init_level(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                n_levels,
    uint                max_allocations,
    uint                block_size,
    uint                data_offset,
    PVDB_INOUT(uint)    offset,
    PVDB_INOUT(uint)    implementation_offset)
{
    if (index == 0u)
        implementation_offset = n_levels * 5;
    const uint begin = (index * 5);
    pvdb_alloc_at(alloc, begin + 0) = 0;
    pvdb_alloc_at(alloc, begin + 1) = max_allocations;
    pvdb_alloc_at(alloc, begin + 2) = block_size;
    pvdb_alloc_at(alloc, begin + 3) = data_offset;
    pvdb_alloc_at(alloc, begin + 4) = implementation_offset;
    implementation_offset += (1u + ((max_allocations - 1u) / 32u));
    offset += 5u;
}


PVDB_INLINE uint
pvdb_allocator_alloc(
    pvdb_buf_inout      alloc,
    uint                index)
{
    const uint first_free_word  = pvdb_alloc_at(alloc, (index * 5) + 0);
    const uint max_allocations  = pvdb_alloc_at(alloc, (index * 5) + 1);
    const uint block_size       = pvdb_alloc_at(alloc, (index * 5) + 2);
    const uint data_offset      = pvdb_alloc_at(alloc, (index * 5) + 3);
    const uint mask_offset      = pvdb_alloc_at(alloc, (index * 5) + 4);
    const uint word_count       = 1u + ((max_allocations - 1) >> 5u);
    uint iter = 0;
    uint word = first_free_word;
    for (;;) {
        PVDB_ASSERT(iter++ < max_allocations && "Allocator is full");
        const uint mask_address     = mask_offset + word;
        const uint current          = pvdb_alloc_at(alloc, mask_address);
        const uint bit_index = findLSB(~current);
        if (bit_index < 32) {
            const uint updated = current | (1u << (bit_index & 31u));
            if (current == atomicCompSwap(pvdb_alloc_at(alloc, mask_address), current, updated)) {
                return data_offset + ((word << 5u) + bit_index) * block_size;
            }
        }
        word = (word + 1) % word_count;
    }
}


PVDB_INLINE void
pvdb_allocator_free(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                ptr)
{
    const uint block_size       = pvdb_alloc_at(alloc, (index * 5) + 2);
    const uint data_offset      = pvdb_alloc_at(alloc, (index * 5) + 3);
    const uint mask_offset      = pvdb_alloc_at(alloc, (index * 5) + 4);
    const uint n                = (ptr - data_offset) / block_size;
    const uint word             = n >> 5u;
    const uint mask_address     = mask_offset + word;
    const uint current          = pvdb_alloc_at(alloc, mask_address);
    const uint updated          = current & ~(1u << (n & 31u));
    atomicExchange(pvdb_alloc_at(alloc, mask_address), updated);
    // update first free word
    atomicMin(pvdb_alloc_at(alloc, (index * 5) + 0), word);
}

//endregion

#else

//region debug allocator

struct pvdb_allocator {
    uint            block_size;
    uint            data_offset;
};


void
pvdb_allocator_init_level(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                n_levels,
    uint                max_allocations,
    uint                block_size,
    uint                data_offset,
    PVDB_INOUT(uint)    offset,
    PVDB_INOUT(uint)    implementation_offset)
{
    pvdb_alloc_at(alloc, 0u) = 0u;
    pvdb_alloc_at(alloc, 1u + (index * 2u)) = block_size;
    pvdb_alloc_at(alloc, 1u + (index * 2u) + 1u) = data_offset;
    implementation_offset += 2u + uint(index == 0u);
    offset += 2u + uint(index == 0u);
}


PVDB_INLINE uint
pvdb_allocator_alloc(
    pvdb_buf_inout      alloc,
    uint                index)
{
    const uint block_size = pvdb_alloc_at(alloc, 1u + (index * 2u));
    const uint data_offset = pvdb_alloc_at(alloc, 1u + (index * 2u) + 1u);
    const uint n = atomicAdd(pvdb_alloc_at(alloc, 0u), block_size);
//    PVDB_PRINTF("\n\t ALLOC: level: %u, block_size: %u, data_offset: %u -> ptr: %u\n", index, block_size, data_offset, data_offset + n * block_size);
    return data_offset + n;
}


PVDB_INLINE void
pvdb_allocator_free(
    pvdb_buf_inout      alloc,
    uint                index,
    uint                ptr)
{}

//endregion

#endif


#endif //PVDB_ALLOCATOR_H
