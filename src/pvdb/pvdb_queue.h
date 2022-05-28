//
// Created by Djordje on 5/26/2022.
//

/// NOTE: This file is intentionally missing header guards so it can be included multiple times

/** MACRO INPUTS
    * PVDB_QUEUE_TYPE           - type of element in queue
    * PVDB_QUEUE_NAME           - name of queue type (e.g. PVDB_QUEUE_TYPE=ivec4, PVDB_QUEUE_NAME=pvdb_queue_ivec4)
    * PVDB_QUEUE_HEADER_ONLY    - define only pvdb_queue_header struct and pvdb_queue_init() function (removes requirement of PVDB_QUEUE_TYPE and PVDB_QUEUE_NAME definitions)
    * PVDB_QUEUE_IMPLEMENTATION - define implementations of queue functions
*/

#include "pvdb_buffer.h"

#ifndef PVDB_QUEUE_HEADER
#define PVDB_QUEUE_HEADER

struct pvdb_queue_header {
    uint             count;
    uint             size;
    uint             indirect_x;
    uint             indirect_y;
    uint             indirect_z;
    uint             PAD[3];
};


PVDB_INLINE void
pvdb_queue_init(
        PVDB_INOUT(pvdb_queue_header) q,
        uint                        size)
{
    q.count = 0u;
    q.size = size;
    q.indirect_x = 0u;
    q.indirect_y = 1u;
    q.indirect_z = 1u;
}

#endif

#ifndef PVDB_QUEUE_HEADER_ONLY

#ifndef PVDB_QUEUE_TYPE
#error Must define PVDB_QUEUE_TYPE
#endif

#ifndef PVDB_QUEUE_NAME
#error Must define PVDB_QUEUE_NAME and it should have a suffix: e.g. pvdb_queue_ivec4   where PVDB_QUEUE_TYPE = ivec4
#endif

#ifdef PVDB_C

struct PVDB_QUEUE_NAME {
    pvdb_queue_header   header;
    PVDB_QUEUE_TYPE     data[];
};

#define pvdb_queue_in                   const PVDB_QUEUE_NAME&
#define pvdb_queue_inout                PVDB_QUEUE_NAME&
#define pvdb_queue_hdr(q)               (q).header
#define pvdb_queue_data(q)              (q).data

#else

#define pvdb_queue_in                   uint
#define pvdb_queue_inout                uint
#define pvdb_queue_hdr(q)               PVDB_GLOBAL_QUEUE[q].header
#define pvdb_queue_data(q)              PVDB_GLOBAL_QUEUE[q].data

#endif

void
pvdb_queue_add(
        pvdb_queue_inout            q,
        PVDB_QUEUE_TYPE             value,
        uint                        local_size);

bool
pvdb_queue_get(
        pvdb_queue_in               q,
        uint                        index,
        PVDB_OUT(PVDB_QUEUE_TYPE)   v);


#ifdef PVDB_QUEUE_IMPLEMENTATION

void
pvdb_queue_add(
        pvdb_queue_inout            q,
        PVDB_QUEUE_TYPE             value,
        uint                        local_size)
{
    const uint index = atomicAdd(pvdb_queue_hdr(q).count, 1u);
    if (index >= pvdb_queue_hdr(q).size) return;
    atomicMax(pvdb_queue_hdr(q).indirect_x, 1u + index / local_size);
    pvdb_queue_data(q)[index] = value;
}


bool
pvdb_queue_get(
        pvdb_queue_in               queue,
        uint                        index,
        PVDB_OUT(PVDB_QUEUE_TYPE)   v)
{
    if (index >= pvdb_queue_hdr(queue).count) return false;
    v = pvdb_queue_data(queue)[index];
    return true;
}

#undef PVDB_QUEUE_IMPLEMENTATION
#undef PVDB_GLOBAL_QUEUE
#undef PVDB_QUEUE_TYPE
#undef PVDB_QUEUE_NAME
#undef pvdb_queue_in
#undef pvdb_queue_inout
#undef pvdb_queue_hdr
#undef pvdb_queue_data

#endif

#endif

#undef PVDB_QUEUE_HEADER_ONLY
