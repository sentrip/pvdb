//
// Created by Djordje on 5/21/2022.
//

/** MACRO INPUTS
    * PVDB_64_BIT           - enable 64 bit numbers/addressing
*/

#ifndef PVDB_BUFFER_H
#define PVDB_BUFFER_H

#include "pvdb_config.h"

#ifdef PVDB_C
#include "pvdb_compat_atomic.h"
#endif

#ifndef PVDB_C

#define pvdb_buf_at(buf, global, addr)          (global[buf].data[addr])

#define pvdb_buf_in                 uint
#define pvdb_buf_out                uint
#define pvdb_buf_inout              uint

#ifdef PVDB_64_BIT

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#define pvdb_buf64_in               uint
#define pvdb_buf64_out              uint
#define pvdb_buf64_inout            uint

#endif

#else

template<typename T>
static inline T& pvdb_buf_at_func(T* mem, size_t i) {
    return mem[i];
}

#define pvdb_buf_at(buf, global, addr)          pvdb_buf_at_func(buf, addr)

typedef const atom_t*               pvdb_buf_in;
typedef atom_t*                     pvdb_buf_out;
typedef atom_t*                     pvdb_buf_inout;

#ifdef PVDB_64_BIT

typedef const atom64_t*             pvdb_buf64_in;
typedef atom64_t*                   pvdb_buf64_out;
typedef atom64_t*                   pvdb_buf64_inout;

#endif

#endif

#endif //PVDB_BUFFER_H
