//
// Created by Djordje on 5/27/2022.
//

#ifndef PVDB_GLOBAL_TEST_H
#define PVDB_GLOBAL_TEST_H

#ifdef PVDB_C

#else

#endif

//region buffer.h

#include "pvdb_config.h"

#define PVDB_DEFINE_BUFFER_READ_WRITE_FUNCTIONS(name) \
    PVDB_INLINE uint name##_read(uint index, uint addr)                         { return PVDB_BUFFER_DATA(name[index])[addr]; } \
    PVDB_INLINE bool name##_read_bit(uint index, uint begin, uint i)            { return (PVDB_BUFFER_DATA(name[index])[begin>>5u] & (1u << (i & 31u))) != 0u; } \
    PVDB_INLINE void name##_write(uint index, uint addr, uint v)                { PVDB_BUFFER_DATA(name[index])[addr] = v; }    \
    PVDB_INLINE uint name##_swap(uint index, uint addr, uint v)                 { return atomicExchange(PVDB_BUFFER_DATA(name[index])[addr], v); } \
    PVDB_INLINE uint name##_cmp_swap(uint index, uint addr, uint cmp, uint v)   { return atomicCompSwap(PVDB_BUFFER_DATA(name[index])[addr], cmp, v); }


#ifdef PVDB_C

typedef uint                    pvdb_buffer;

#define PVDB_BUFFER_DATA(v) v

#define PVDB_DEFINE_BUFFER(name, count, ...) \
    extern atom_t* name[count]

#define PVDB_DEFINE_BUFFER_STORAGE(name) \
    atom_t* name[sizeof(name)/sizeof(name[0])]

#else

#define pvdb_buffer     uint

#define PVDB_BUFFER_DATA(v) (v).data

#define PVDB_DEFINE_BUFFER(name, count, bind, ...) \
    layout(std430, bind) __VA_ARGS__ buffer name##_t { uint data[]; } name[count]

#endif

/// atomic compatibility
#ifdef PVDB_C
#include "pvdb_compat_atomic.h"
#endif

//endregion

//region user code

const uint LOG2DIM[4] = {1,2,3,4};

PVDB_DEFINE_BUFFER(global_buffer, 1, binding = 0);
PVDB_DEFINE_BUFFER_READ_WRITE_FUNCTIONS(global_buffer)

PVDB_INLINE uint global_read(pvdb_buffer buf, uint addr) { return global_buffer_read(buf, addr); }
PVDB_INLINE uint global_log2dim(uint level) { return LOG2DIM[level]; }

//endregion

#endif //PVDB_GLOBAL_TEST_H
