//
// Created by Djordje on 5/27/2022.
//

#ifndef PVDB_COMPAT_ATOMIC_H
#define PVDB_COMPAT_ATOMIC_H

#include <atomic>

typedef std::atomic_uint32_t        atom_t;
//typedef uint                        atom_t;
typedef const atom_t*               pvdb_buf_in;
typedef atom_t*                     pvdb_buf_out;
typedef atom_t*                     pvdb_buf_inout;

template<uint N>
struct pvdb_buf_t {
    atom_t data[N]{};

    uint    operator[](uint i) const { return data[i]; }
    atom_t& operator[](uint i)       { return data[i]; }

    operator const uint*()     const { return &data[0]; }
    operator atom_t*()               { return &data[0]; }

    template<uint Q>
    void set(const pvdb_buf_t<Q>& v, uint dst_offset = 0, uint src_offset = 0, uint src_size = Q) {
        for (uint i = 0; i < src_size; ++i) data[dst_offset+i] = v[src_offset + i];
    }
};

#ifdef PVDB_64_BIT

typedef uint64_t                    uint64;
typedef std::atomic_uint64_t        atom64_t;
//typedef uint64                      atom64_t;
typedef const atom64_t*             pvdb_buf64_in;
typedef atom64_t*                   pvdb_buf64_out;
typedef atom64_t*                   pvdb_buf64_inout;

template<uint N>
struct pvdb_buf64_t {
    atom64_t data[N]{};

    uint64    operator[](uint64 i) const { return data[i]; }
    atom64_t& operator[](uint64 i)       { return data[i]; }

    operator const uint64*()       const { return &data[0]; }
    operator atom64_t*()                 { return &data[0]; }

    template<uint Q>
    void set(const pvdb_buf64_t<Q>& v, uint dst_offset = 0, uint src_offset = 0, uint src_size = Q) {
        for (uint i = 0; i < src_size; ++i) data[dst_offset+i] = v[src_offset + i];
    }
};

#endif


template<typename A, typename U>
inline U atomicAdd(A& mem, U n)
{
    const U prev = mem;
    mem += n;
    return prev;
}

template<typename A, typename U>
inline U atomicCompSwap(A& mem, U cmp, U value)
{
    if constexpr(std::is_same_v<A, std::atomic_uint32_t> || std::is_same_v<A, std::atomic_uint64_t>) {
        mem.compare_exchange_strong(cmp, value);
        return cmp;
    } else {
        const U prev = mem;
        if (mem == cmp) mem = value;
        return prev;
    }
}

template<typename A, typename U>
inline U atomicExchange(A& mem, U v)
{
    if constexpr(std::is_same_v<A, std::atomic_uint32_t> || std::is_same_v<A, std::atomic_uint64_t>) {
        return mem.exchange(v);
    } else {
        const U prev = mem;
        mem = v;
        return prev;
    }
}

template<typename A, typename U>
inline U atomicMin(A& mem, U v)
{
    const U prev = mem;
    mem = v < prev ? v : prev;
    return prev;
}

template<typename A, typename U>
inline U atomicMax(A& mem, U v)
{
    const U prev = mem;
    mem = v < prev ? prev : v;
    return prev;
}

template<typename A, typename U>
inline U atomicOr(A& mem, U v)
{
    const U prev = mem;
    mem |= v;
    return prev;
}

template<typename A, typename U>
inline U atomicAnd(A& mem, U v)
{
    const U prev = mem;
    mem &= v;
    return prev;
}

#endif //PVDB_COMPAT_ATOMIC_H
