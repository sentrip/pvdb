//
// Created by Djordje on 5/21/2022.
//

/** MACRO INPUTS
    * PVDB_C                - whether compiling in c++ or glsl
    * PVDB_ENABLE_PRINTF    - whether or not to enable printing
*/

#ifndef PVDB_CONFIG_H
#define PVDB_CONFIG_H

//#define PVDB_USE_IMAGES

/// constants
#define PVDB_INT_MAX                0x7FFFFFFF
#define PVDB_UINT_MAX               0xFFFFFFFFu
#define PVDB_EPS                    3.552713678800501e-15f
#define PVDB_PI                     3.14159265358979323846264338327950288


/// basic stack
#define pvdb_stack_t(name, T, N)    uint name##_size = 0; uint name##_max = N; T name[N]
#define pvdb_stack_empty(s)         (s##_size == 0)
#ifdef PVDB_C
#define pvdb_stack_push(s, v)       (s[s##_size++] = (v), PVDB_ASSERT(s##_size <= s##_max && "Stack is full - cannot push"))
#define pvdb_stack_pop(s)           (PVDB_ASSERT(s##_size > 0 && "Stack is empty - cannot pop"), s[--s##_size])
#else
#define pvdb_stack_push(s, v)       s[s##_size++] = (v)
#define pvdb_stack_pop(s)           s[--s##_size]
#endif


/// ivec3 <-> index
#define pvdb_coord_to_index(p, l2d)                 (((p).x << int(2u * (l2d))) + ((p).y << int(l2d)) + (p).z)
#define pvdb_index_to_coord_x(i, l2d)               int((i) >> (2u * (l2d)))
#define pvdb_index_to_coord_y(i, l2d)               int(((i) >> (l2d)) & ((1u << (l2d))-1u))
#define pvdb_index_to_coord_z(i, l2d)               int((i) & ((1u << (l2d))-1u))
#define pvdb_index_to_coord(i, l2d)                 ivec3(pvdb_index_to_coord_x(i, l2d), pvdb_index_to_coord_y(i, l2d), pvdb_index_to_coord_z(i, l2d))


/// GPU global bindings
#define PVDB_BINDING_TREE               0
#define PVDB_BINDING_TREE_ALLOC         1
#ifdef PVDB_USE_IMAGES
#define PVDB_BINDING_TREE_ATLAS         2
#define PVDB_BINDING_CAMERA             3
#else
#define PVDB_BINDING_CAMERA             2
#endif

/// platform compatibility
#ifndef PVDB_C

#define PVDB_INLINE

#define PVDB_IN(X)                  in X
#define PVDB_OUT(X)                 out X
#define PVDB_INOUT(X)               inout X
#define PVDB_ARRAY_IN(X, n, N)      in X[N] n
#define PVDB_ARRAY_OUT(X, n, N)     out X[N] n
#define PVDB_ARRAY_INOUT(X, n, N)   inout X[N] n

#define PVDB_BARRIER()              barrier()

#define PVDB_ASSERT(...)

#ifdef PVDB_ENABLE_PRINTF
#define PVDB_PRINTF(...)            debugPrintfEXT(__VA_ARGS__)
#else
#define PVDB_PRINTF(...)
#endif

#else

#include <cassert>

#define PVDB_INLINE                 static inline
#define PVDB_IN(X)                  const X&
#define PVDB_OUT(X)                 X&
#define PVDB_INOUT(X)               X&
#define PVDB_ARRAY_IN(T, n, N)      const T (&(n))[N]
#define PVDB_ARRAY_OUT(T, n, N)     T (&(n))[N]
#define PVDB_ARRAY_INOUT(T, n, N)   T (&(n))[N]

#define PVDB_BARRIER()

#define PVDB_ASSERT(...)            assert(__VA_ARGS__)

#ifdef PVDB_ENABLE_PRINTF
#include <cstdio>
#define PVDB_PRINTF(...)            (printf(__VA_ARGS__), fflush(stdout))
#else
#define PVDB_PRINTF(...)
#endif

typedef unsigned int        uint;

#endif

/// c-implementation
#ifdef PVDB_C

inline uint    floatBitsToUint(float f) { return uint((uint&)f); }
inline float   uintBitsToFloat(uint u)  { return float((float&)u); }
constexpr uint findLSB(uint v) { for(uint i = 0; i < 32; ++i) if ((v & (1u << i)) != 0u) return i; return -1u; }

#include "pvdb_math_compat.h"

#endif

PVDB_INLINE bool is_power_of_two(uint x) { return (x != 0u) && (x & (x - 1u)) == 0u; }

#endif //PVDB_CONFIG_H
