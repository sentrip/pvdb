//
// Created by Djordje on 5/21/2022.
//

#ifndef PVDB_COMPAT_H
#define PVDB_COMPAT_H

#include <cmath>

struct ivec3 {
    int x{}, y{}, z{};
    constexpr ivec3() = default;
    constexpr ivec3(int x) : x{x}, y{x}, z{x} {}
    constexpr ivec3(int x, int y, int z) : x{x}, y{y}, z{z} {}
    template<typename V> explicit constexpr ivec3(const V& v) : x{int(v.x)}, y{int(v.y)}, z{int(v.z)} {}

    constexpr int&         operator[](uint i)                 { return ((int*)this)[i]; }
    constexpr int          operator[](uint i)           const { return ((const int*)this)[i]; }
    constexpr ivec3        operator- ()                 const { return {-x, -y, -z}; }
    constexpr ivec3        operator+ ()                 const { return {+x, +y, +z}; }

    constexpr ivec3&       operator+=(int v)                  { x += v; y += v; z += v; return *this; }
    constexpr ivec3&       operator-=(int v)                  { x -= v; y -= v; z -= v; return *this; }
    constexpr ivec3&       operator*=(int v)                  { x *= v; y *= v; z *= v; return *this; }
    constexpr ivec3&       operator/=(int v)                  { x /= v; y /= v; z /= v; return *this; }
    constexpr ivec3        operator+ (int v)            const { return {x + v, y + v, z + v}; }
    constexpr ivec3        operator- (int v)            const { return {x - v, y - v, z - v}; }
    constexpr ivec3        operator* (int v)            const { return {x * v, y * v, z * v}; }
    constexpr ivec3        operator/ (int v)            const { return {x / v, y / v, z / v}; }

    constexpr ivec3&       operator+=(const ivec3& o)         { x += o.x; y += o.y; z += o.z; return *this; }
    constexpr ivec3&       operator-=(const ivec3& o)         { x -= o.x; y -= o.y; z -= o.z; return *this; }
    constexpr ivec3&       operator*=(const ivec3& o)         { x *= o.x; y *= o.y; z *= o.z; return *this; }
    constexpr ivec3&       operator/=(const ivec3& o)         { x /= o.x; y /= o.y; z /= o.z; return *this; }
    constexpr ivec3        operator+ (const ivec3& o)   const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr ivec3        operator- (const ivec3& o)   const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr ivec3        operator* (const ivec3& o)   const { return {x * o.x, y * o.y, z * o.z}; }
    constexpr ivec3        operator/ (const ivec3& o)   const { return {x / o.x, y / o.y, z / o.z}; }

    constexpr ivec3        operator~ ()                 const { return {~x, ~y, ~z}; }
    constexpr ivec3&       operator&= (int s)                 { x &= s; y &= s; z &= s; return *this; }
    constexpr ivec3&       operator<<=(int s)                 { x <<= s; y <<= s; z <<= s; return *this; }
    constexpr ivec3&       operator>>=(int s)                 { x >>= s; y >>= s; z >>= s; return *this; }
    constexpr ivec3        operator& (int s)            const { return {x & s, y & s, z & s}; }
    constexpr ivec3        operator<<(int s)            const { return {x << s, y << s, z << s}; }
    constexpr ivec3        operator>>(int s)            const { return {x >> s, y >> s, z >> s}; }
    constexpr ivec3&       operator&= (const ivec3& o)        { x &= o.x; y &= o.y; z &= o.z; return *this; }
    constexpr ivec3&       operator<<=(const ivec3& o)        { x <<= o.x; y <<= o.y; z <<= o.z; return *this; }
    constexpr ivec3&       operator>>=(const ivec3& o)        { x >>= o.x; y >>= o.y; z >>= o.z; return *this; }
    constexpr ivec3        operator& (const ivec3& o)   const { return {x & o.x, y & o.y, z & o.z}; }
    constexpr ivec3        operator<<(const ivec3& o)   const { return {x << o.x, y << o.y, z << o.z}; }
    constexpr ivec3        operator>>(const ivec3& o)   const { return {x >> o.x, y >> o.y, z >> o.z}; }

    constexpr ivec3        operator==(const ivec3& o)   const { return {int(x == o.x), int(y == o.y), int(z == o.z)}; }
    constexpr ivec3        operator!=(const ivec3& o)   const { return {int(x != o.x), int(y != o.y), int(z != o.z)}; }
//    constexpr ivec3        operator< (const ivec3& o)   const { return {int(x < o.x), int(y < o.y), int(z < o.z)}; }
//    constexpr ivec3        operator> (const ivec3& o)   const { return {int(x > o.x), int(y > o.y), int(z > o.z)}; }
//    constexpr ivec3        operator<=(const ivec3& o)   const { return {int(x <= o.x), int(y <= o.y), int(z <= o.z)}; }
//    constexpr ivec3        operator>=(const ivec3& o)   const { return {int(x >= o.x), int(y >= o.y), int(z >= o.z)}; }
};

struct ivec4 {
    union {
        struct {
            int x, y, z;
        };
        struct {
            ivec3 xyz;
        };
    };
    int w;

    constexpr ivec4() : x{}, y{}, z{}, w{} {}
    constexpr ivec4(int x) : x{x}, y{x}, z{x}, w{x} {}
    constexpr ivec4(int x, int y, int z, int w) : x{x}, y{y}, z{z}, w{w} {}
    constexpr ivec4(const ivec3& xyz, int w) : x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {}
    template<typename V> explicit constexpr ivec4(const V& v) : x{int(v.x)}, y{int(v.y)}, z{int(v.z)}, w{int(v.w)} {}
    constexpr operator ivec3() const { return {x, y, z}; }

    constexpr int&         operator[](uint i)                 { return ((int*)this)[i]; }
    constexpr int          operator[](uint i)           const { return ((const int*)this)[i]; }
    constexpr ivec4        operator- ()                 const { return {-x, -y, -z, -w}; }
    constexpr ivec4        operator+ ()                 const { return {+x, +y, +z, +w}; }

    constexpr ivec4&       operator+=(int v)                  { x += v; y += v; z += v; w += v; return *this; }
    constexpr ivec4&       operator-=(int v)                  { x -= v; y -= v; z -= v; w -= v; return *this; }
    constexpr ivec4&       operator*=(int v)                  { x *= v; y *= v; z *= v; w *= v; return *this; }
    constexpr ivec4&       operator/=(int v)                  { x /= v; y /= v; z /= v; w /= v; return *this; }
    constexpr ivec4        operator+ (int v)            const { return {x + v, y + v, z + v, w + v}; }
    constexpr ivec4        operator- (int v)            const { return {x - v, y - v, z - v, w - v}; }
    constexpr ivec4        operator* (int v)            const { return {x * v, y * v, z * v, w * v}; }
    constexpr ivec4        operator/ (int v)            const { return {x / v, y / v, z / v, w / v}; }

    constexpr ivec4&       operator+=(const ivec4& o)         { x += o.x; y += o.y; z += o.z; w += o.w; return *this; }
    constexpr ivec4&       operator-=(const ivec4& o)         { x -= o.x; y -= o.y; z -= o.z; w -= o.w; return *this; }
    constexpr ivec4&       operator*=(const ivec4& o)         { x *= o.x; y *= o.y; z *= o.z; w *= o.w; return *this; }
    constexpr ivec4&       operator/=(const ivec4& o)         { x /= o.x; y /= o.y; z /= o.z; w /= o.w; return *this; }
    constexpr ivec4        operator+ (const ivec4& o)   const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    constexpr ivec4        operator- (const ivec4& o)   const { return {x - o.x, y - o.y, z - o.z, w - o.w}; }
    constexpr ivec4        operator* (const ivec4& o)   const { return {x * o.x, y * o.y, z * o.z, w * o.w}; }
    constexpr ivec4        operator/ (const ivec4& o)   const { return {x / o.x, y / o.y, z / o.z, w / o.w}; }

    constexpr ivec4        operator~ ()                 const { return {~x, ~y, ~z, ~w}; }
    constexpr ivec4&       operator&= (int s)                 { x &= s; y &= s; z &= s; w &= s; return *this; }
    constexpr ivec4&       operator<<=(int s)                 { x <<= s; y <<= s; z <<= s; w <<= s; return *this; }
    constexpr ivec4&       operator>>=(int s)                 { x >>= s; y >>= s; z >>= s; w >>= s; return *this; }
    constexpr ivec4        operator& (int s)            const { return {x & s, y & s, z & s, w & s}; }
    constexpr ivec4        operator<<(int s)            const { return {x << s, y << s, z << s, w << s}; }
    constexpr ivec4        operator>>(int s)            const { return {x >> s, y >> s, z >> s, w >> s}; }
    constexpr ivec4&       operator&= (const ivec4& o)        { x &= o.x; y &= o.y; z &= o.z; w &= o.w; return *this; }
    constexpr ivec4&       operator<<=(const ivec4& o)        { x <<= o.x; y <<= o.y; z <<= o.z; w <<= o.w; return *this; }
    constexpr ivec4&       operator>>=(const ivec4& o)        { x >>= o.x; y >>= o.y; z >>= o.z; w >>= o.w; return *this; }
    constexpr ivec4        operator& (const ivec4& o)   const { return {x & o.x, y & o.y, z & o.z, w & o.w}; }
    constexpr ivec4        operator<<(const ivec4& o)   const { return {x << o.x, y << o.y, z << o.z, w << o.w}; }
    constexpr ivec4        operator>>(const ivec4& o)   const { return {x >> o.x, y >> o.y, z >> o.z, w >> o.w}; }

    constexpr ivec4        operator==(const ivec4& o)   const { return {int(x == o.x), int(y == o.y), int(z == o.z), int(w == o.w)}; }
    constexpr ivec4        operator!=(const ivec4& o)   const { return {int(x != o.x), int(y != o.y), int(z != o.z), int(w != o.w)}; }
//    constexpr ivec4        operator< (const ivec4& o)   const { return {int(x < o.x), int(y < o.y), int(z < o.z), int(w < o.w)}; }
//    constexpr ivec4        operator> (const ivec4& o)   const { return {int(x > o.x), int(y > o.y), int(z > o.z), int(w > o.w)}; }
//    constexpr ivec4        operator<=(const ivec4& o)   const { return {int(x <= o.x), int(y <= o.y), int(z <= o.z), int(w <= o.w)}; }
//    constexpr ivec4        operator>=(const ivec4& o)   const { return {int(x >= o.x), int(y >= o.y), int(z >= o.z), int(w >= o.w)}; }
};

struct uvec4 {
    uint x, y, z, w;
    constexpr uvec4() : x{}, y{}, z{}, w{} {}
    constexpr uvec4(uint x) : x{x}, y{x}, z{x}, w{x} {}
    constexpr uvec4(uint x, uint y, uint z, uint w) : x{x}, y{y}, z{z}, w{w} {}
    template<typename V> explicit constexpr uvec4(const V& v) : x{int(v.x)}, y{int(v.y)}, z{int(v.z)}, w{int(v.w)} {}

    constexpr uint&        operator[](uint i)                 { return ((uint*)this)[i]; }
    constexpr uint          operator[](uint i)          const { return ((const uint*)this)[i]; }
};


struct vec2 {
    float x{}, y{};

    constexpr vec2() = default;
    constexpr vec2(float v) : x{v}, y{v} {}
    constexpr vec2(float x, float y) : x{x}, y{y} {}
    template<typename V> explicit constexpr vec2(const V& v) : x{float(v.x)}, y{float(v.y)} {}

};

struct vec3 {
    float x{}, y{}, z{};
    constexpr vec3() = default;
    constexpr vec3(float x) : x{x}, y{x}, z{x} {}
    constexpr vec3(float x, float y, float z) : x{x}, y{y}, z{z} {}
    template<typename V> explicit constexpr vec3(const V& v) : x{float(v.x)}, y{float(v.y)}, z{float(v.z)} {}

    constexpr float&       operator[](uint i)                 { return ((float*)this)[i]; }
    constexpr float        operator[](uint i)           const { return ((const float*)this)[i]; }
    constexpr vec3         operator- ()                 const { return {-x, -y, -z}; }
    constexpr vec3         operator+ ()                 const { return {+x, +y, +z}; }

    constexpr vec3&        operator+=(float v)                { x += v; y += v; z += v; return *this; }
    constexpr vec3&        operator-=(float v)                { x -= v; y -= v; z -= v; return *this; }
    constexpr vec3&        operator*=(float v)                { x *= v; y *= v; z *= v; return *this; }
    constexpr vec3&        operator/=(float v)                { x /= v; y /= v; z /= v; return *this; }
    constexpr vec3         operator+ (float v)          const { return {x + v, y + v, z + v}; }
    constexpr vec3         operator- (float v)          const { return {x - v, y - v, z - v}; }
    constexpr vec3         operator* (float v)          const { return {x * v, y * v, z * v}; }
    constexpr vec3         operator/ (float v)          const { return {x / v, y / v, z / v}; }

    constexpr vec3&        operator+=(const vec3& o)          { x += o.x; y += o.y; z += o.z; return *this; }
    constexpr vec3&        operator-=(const vec3& o)          { x -= o.x; y -= o.y; z -= o.z; return *this; }
    constexpr vec3&        operator*=(const vec3& o)          { x *= o.x; y *= o.y; z *= o.z; return *this; }
    constexpr vec3&        operator/=(const vec3& o)          { x /= o.x; y /= o.y; z /= o.z; return *this; }
    constexpr vec3         operator+ (const vec3& o)    const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr vec3         operator- (const vec3& o)    const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr vec3         operator* (const vec3& o)    const { return {x * o.x, y * o.y, z * o.z}; }
    constexpr vec3         operator/ (const vec3& o)    const { return {x / o.x, y / o.y, z / o.z}; }

    /// NOTE: This differs from glsl, but to avoid defining a bvec3 struct ivec3 is used for bvec3
    constexpr ivec3        operator==(const vec3& o)    const { return {int(x == o.x), int(y == o.y), int(z == o.z)}; }
    constexpr ivec3        operator!=(const vec3& o)    const { return {int(x != o.x), int(y != o.y), int(z != o.z)}; }
    constexpr ivec3        operator< (const vec3& o)    const { return {int(x < o.x), int(y < o.y), int(z < o.z)}; }
    constexpr ivec3        operator> (const vec3& o)    const { return {int(x > o.x), int(y > o.y), int(z > o.z)}; }
    constexpr ivec3        operator<=(const vec3& o)    const { return {int(x <= o.x), int(y <= o.y), int(z <= o.z)}; }
    constexpr ivec3        operator>=(const vec3& o)    const { return {int(x >= o.x), int(y >= o.y), int(z >= o.z)}; }
};

struct vec4 {
    union {
        struct {
            float x, y, z;
        };
        struct {
            vec3 xyz;
        };
    };
    float w{};

    constexpr vec4() : x{}, y{}, z{}, w{} {}
    constexpr vec4(float x) : x{x}, y{x}, z{x}, w{x} {}
    constexpr vec4(float x, float y, float z, float w) : x{x}, y{y}, z{z}, w{w} {}
    constexpr vec4(const vec3& xyz, float w) : x{xyz.x}, y{xyz.y}, z{xyz.z}, w{w} {}
    template<typename V> explicit constexpr vec4(const V& v) : x{float(v.x)}, y{float(v.y)}, z{float(v.z)}, w{float(v.w)} {}
    constexpr operator vec3() const { return {x, y, z}; }

    constexpr float&       operator[](uint i)                 { return ((float*)this)[i]; }
    constexpr float        operator[](uint i)           const { return ((const float*)this)[i]; }
    constexpr vec4         operator- ()                 const { return {-x, -y, -z, -w}; }
    constexpr vec4         operator+ ()                 const { return {+x, +y, +z, +w}; }

    constexpr vec4&        operator+=(float v)                { x += v; y += v; z += v; w += v; return *this; }
    constexpr vec4&        operator-=(float v)                { x -= v; y -= v; z -= v; w -= v; return *this; }
    constexpr vec4&        operator*=(float v)                { x *= v; y *= v; z *= v; w *= v; return *this; }
    constexpr vec4&        operator/=(float v)                { x /= v; y /= v; z /= v; w /= v; return *this; }
    constexpr vec4         operator+ (float v)          const { return {x + v, y + v, z + v, w + v}; }
    constexpr vec4         operator- (float v)          const { return {x - v, y - v, z - v, w - v}; }
    constexpr vec4         operator* (float v)          const { return {x * v, y * v, z * v, w * v}; }
    constexpr vec4         operator/ (float v)          const { return {x / v, y / v, z / v, w / v}; }

    constexpr vec4&        operator+=(const vec4& o)          { x += o.x; y += o.y; z += o.z; w += o.w; return *this; }
    constexpr vec4&        operator-=(const vec4& o)          { x -= o.x; y -= o.y; z -= o.z; w -= o.w; return *this; }
    constexpr vec4&        operator*=(const vec4& o)          { x *= o.x; y *= o.y; z *= o.z; w *= o.w; return *this; }
    constexpr vec4&        operator/=(const vec4& o)          { x /= o.x; y /= o.y; z /= o.z; w /= o.w; return *this; }
    constexpr vec4         operator+ (const vec4& o)    const { return {x + o.x, y + o.y, z + o.z, w + o.w}; }
    constexpr vec4         operator- (const vec4& o)    const { return {x - o.x, y - o.y, z - o.z, w - o.w}; }
    constexpr vec4         operator* (const vec4& o)    const { return {x * o.x, y * o.y, z * o.z, w * o.w}; }
    constexpr vec4         operator/ (const vec4& o)    const { return {x / o.x, y / o.y, z / o.z, w / o.w}; }

    constexpr ivec4        operator==(const vec4& o)    const { return {int(x == o.x), int(y == o.y), int(z == o.z), int(w == o.w)}; }
    constexpr ivec4        operator!=(const vec4& o)    const { return {int(x != o.x), int(y != o.y), int(z != o.z), int(w != o.w)}; }
    constexpr ivec4        operator< (const vec4& o)    const { return {int(x < o.x), int(y < o.y), int(z < o.z), int(w < o.w)}; }
    constexpr ivec4        operator> (const vec4& o)    const { return {int(x > o.x), int(y > o.y), int(z > o.z), int(w > o.w)}; }
    constexpr ivec4        operator<=(const vec4& o)    const { return {int(x <= o.x), int(y <= o.y), int(z <= o.z), int(w <= o.w)}; }
    constexpr ivec4        operator>=(const vec4& o)    const { return {int(x >= o.x), int(y >= o.y), int(z >= o.z), int(w >= o.w)}; }
};


constexpr     bool      any(ivec3 v)                        { return v.x > 0 || v.y > 0 || v.z > 0; }
constexpr     bool      all(ivec3 v)                        { return v.x > 0 && v.y > 0 && v.z > 0; }
constexpr     float     sign(float v)                       { return float(int(0.0f < v) - int(v < 0.0f)); }
constexpr     vec3      sign(vec3 v)                        { return {sign(v.x), sign(v.y), sign(v.z)}; }
constexpr     uint      min(uint l, uint r)                 { return l < r ? l : r; }
constexpr     uint      max(uint l, uint r)                 { return l < r ? r : l; }
static inline float     min(float l, float r)               { return fminf(l, r); }
static inline float     max(float l, float r)               { return fmaxf(l, r); }
static inline vec3      min(vec3 l, vec3 r)                 { return {fminf(l.x, r.x), fminf(l.y, r.y), fminf(l.z, r.z)}; }
static inline vec3      abs(vec3 v)                         { return {fabsf(v.x), fabsf(v.y), fabsf(v.z)}; }
static inline vec3      floor(vec3 v)                       { return {floorf(v.x), floorf(v.y), floorf(v.z)}; }
static inline vec3      max(vec3 l, vec3 r)                 { return {max(l.x, r.x), max(l.y, r.y), max(l.z, r.z)}; }
static inline vec3      clamp(vec3 l, float mn, float mx)   { return {max(min(l.x, mx), mn), max(min(l.y, mx), mn), max(min(l.z, mx), mn)}; }
static inline float     length(const vec3& v)               { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline vec3      normalize(const vec3& v)            { return v / length(v); }
constexpr     float     dot(const vec4& l, const vec4& r)   { float v = 0.0f; for (int i = 0; i < 4; ++i) { v += l[i] * r[i]; } return v; }
constexpr     vec3      cross(const vec3& a, const vec3& b) { return {a.y*b.z - b.y*a.z, a.z*b.x - b.z*a.x, a.x*b.y - b.x*a.y}; }


struct mat4 {
    vec4 cols[4];

    constexpr mat4() = default;
    constexpr mat4(vec4 c0, vec4 c1, vec4 c2, vec4 c3) : cols{c0, c1, c2, c3} {}

    constexpr vec4&       operator[](uint i)       { return cols[i]; }
    constexpr const vec4& operator[](uint i) const { return cols[i]; }
    constexpr vec4 row(uint i)        const { return {cols[0][i], cols[1][i], cols[2][i], cols[3][i]}; }

    mat4 operator*(const mat4& o) const {
        mat4 out{};
        for(int col = 0; col != 4; ++col)
            for(int row = 0; row != 4; ++row)
                for(int pos = 0; pos != 4; ++pos)
                    out.cols[col][row] += cols[pos][row]*o.cols[col][pos];
        return out;
    }

    vec4 operator*(const vec4& v) const
    {
        vec4 res = vec4(0.0f);
        for (int r = 0; r < 4; ++r) res[r] = dot(row(r), v);
        return res;
    }

};

#endif //PVDB_COMPAT_H
