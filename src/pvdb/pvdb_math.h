//
// Created by Djordje on 5/21/2022.
//

#ifndef PVDB_MATH_H
#define PVDB_MATH_H

#include "pvdb_config.h"

//region general

#define pvdb_rad_to_deg(r) ((r) * 180.0f / float(PVDB_PI))
#define pvdb_deg_to_rad(d) ((d) * float(PVDB_PI) / 180.0f)

PVDB_INLINE float
pvdb_copysign(
    float               n,
    float               sign)
{
    const uint nu = floatBitsToUint(n) & PVDB_INT_MAX;
    const uint su = floatBitsToUint(sign) & ~PVDB_INT_MAX;
    return uintBitsToFloat(nu | su);
}


PVDB_INLINE
float pvdb_min(
    vec3                 v)
{
    return min(min(v.x, v.y), v.z);
}


PVDB_INLINE
float pvdb_max(
    vec3                 v)
{
    return max(max(v.x, v.y), v.z);
}

//endregion

//region mat4

#define pvdb_mat4_row(m, row) vec4((m)[0][row], (m)[1][row], (m)[2][row], (m)[3][row])


PVDB_INLINE void
pvdb_mat4_projection_orthographic(
    PVDB_OUT(mat4)      m,
    PVDB_IN(vec2)       size,
    float               near,
    float               far)
{
    const float z_scale = 2.0f / (near - far);
    m = mat4(vec4(2.0f / size.x,          0.0f,              0.0f, 0.0f),
             vec4(         0.0f, 2.0f / size.y,              0.0f, 0.0f),
             vec4(         0.0f,          0.0f,           z_scale, 0.0f),
             vec4(         0.0f,          0.0f, near*z_scale-1.0f, 1.0f));
}


PVDB_INLINE void
pvdb_mat4_projection_perspective(
    PVDB_OUT(mat4)      m,
    PVDB_IN(vec2)       size,
    float               near,
    float               far)
{
    const float z_scale = 1.0f / (near - far);
    const float m22 = (far + near) * z_scale;
    const float m32 = 2.0f * far * near * z_scale;
    m = mat4(vec4(2.0f * near / size.x,                 0.0f, 0.0f,  0.0f),
             vec4(                0.0f, 2.0f * near / size.y, 0.0f,  0.0f),
             vec4(                0.0f,                 0.0f,  m22,  -1.0f),
             vec4(                0.0f,                 0.0f,  m32,   0.0f));
}


PVDB_INLINE void
pvdb_mat4_projection_perspective(
    PVDB_OUT(mat4)      m,
    float               fov,
    float               aspect_ratio,
    float               near,
    float               far)
{
    const float f = 2.0f * near * tan(fov*0.5f);
    pvdb_mat4_projection_perspective(m, vec2(f, f * (1.0f / aspect_ratio)), near, far);
}


PVDB_INLINE void
pvdb_mat4_look_at(
    PVDB_OUT(mat4)      m,
    PVDB_IN(vec3)       eye,
    PVDB_IN(vec3)       target,
    PVDB_IN(vec3)       up)
{
    const vec3 backward = normalize(eye - target);
    const vec3 right = normalize(cross(up, backward));
    const vec3 real_up = cross(backward, right);
    m = mat4(vec4(right, 0.0f),
             vec4(real_up, 0.0f),
             vec4(backward, 0.0f),
             vec4(eye, 1.0f));
}


PVDB_INLINE mat4
pvdb_mat4_inverted(
    PVDB_IN(mat4)       m)
{
    const float A2323 = m[2][2] * m[3][3] - m[2][3] * m[3][2] ;
    const float A1323 = m[2][1] * m[3][3] - m[2][3] * m[3][1] ;
    const float A1223 = m[2][1] * m[3][2] - m[2][2] * m[3][1] ;
    const float A0323 = m[2][0] * m[3][3] - m[2][3] * m[3][0] ;
    const float A0223 = m[2][0] * m[3][2] - m[2][2] * m[3][0] ;
    const float A0123 = m[2][0] * m[3][1] - m[2][1] * m[3][0] ;
    const float A2313 = m[1][2] * m[3][3] - m[1][3] * m[3][2] ;
    const float A1313 = m[1][1] * m[3][3] - m[1][3] * m[3][1] ;
    const float A1213 = m[1][1] * m[3][2] - m[1][2] * m[3][1] ;
    const float A2312 = m[1][2] * m[2][3] - m[1][3] * m[2][2] ;
    const float A1312 = m[1][1] * m[2][3] - m[1][3] * m[2][1] ;
    const float A1212 = m[1][1] * m[2][2] - m[1][2] * m[2][1] ;
    const float A0313 = m[1][0] * m[3][3] - m[1][3] * m[3][0] ;
    const float A0213 = m[1][0] * m[3][2] - m[1][2] * m[3][0] ;
    const float A0312 = m[1][0] * m[2][3] - m[1][3] * m[2][0] ;
    const float A0212 = m[1][0] * m[2][2] - m[1][2] * m[2][0] ;
    const float A0113 = m[1][0] * m[3][1] - m[1][1] * m[3][0] ;
    const float A0112 = m[1][0] * m[2][1] - m[1][1] * m[2][0] ;

    float det = m[0][0] * ( m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223 )
        - m[0][1] * ( m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223 )
        + m[0][2] * ( m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123 )
        - m[0][3] * ( m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123 ) ;
    det = 1.0f / det;

    return mat4(
       vec4(
           det *   ( m[1][1] * A2323 - m[1][2] * A1323 + m[1][3] * A1223 ),
           det * - ( m[0][1] * A2323 - m[0][2] * A1323 + m[0][3] * A1223 ),
           det *   ( m[0][1] * A2313 - m[0][2] * A1313 + m[0][3] * A1213 ),
           det * - ( m[0][1] * A2312 - m[0][2] * A1312 + m[0][3] * A1212 )
       ),
       vec4(
           det * - ( m[1][0] * A2323 - m[1][2] * A0323 + m[1][3] * A0223 ),
           det *   ( m[0][0] * A2323 - m[0][2] * A0323 + m[0][3] * A0223 ),
           det * - ( m[0][0] * A2313 - m[0][2] * A0313 + m[0][3] * A0213 ),
           det *   ( m[0][0] * A2312 - m[0][2] * A0312 + m[0][3] * A0212 )
       ),
       vec4(
           det *   ( m[1][0] * A1323 - m[1][1] * A0323 + m[1][3] * A0123 ),
           det * - ( m[0][0] * A1323 - m[0][1] * A0323 + m[0][3] * A0123 ),
           det *   ( m[0][0] * A1313 - m[0][1] * A0313 + m[0][3] * A0113 ),
           det * - ( m[0][0] * A1312 - m[0][1] * A0312 + m[0][3] * A0112 )
       ),
       vec4(
           det * - ( m[1][0] * A1223 - m[1][1] * A0223 + m[1][2] * A0123 ),
           det *   ( m[0][0] * A1223 - m[0][1] * A0223 + m[0][2] * A0123 ),
           det * - ( m[0][0] * A1213 - m[0][1] * A0213 + m[0][2] * A0113 ),
           det *   ( m[0][0] * A1212 - m[0][1] * A0212 + m[0][2] * A0112 )
       )
   );
}


PVDB_INLINE void
pvdb_mat4_to_frustum(
    PVDB_IN(mat4)        m,
    PVDB_ARRAY_OUT(vec4, frustum, 6))
{
    frustum[0] = pvdb_mat4_row(m, 3) + pvdb_mat4_row(m, 0);
    frustum[1] = pvdb_mat4_row(m, 3) - pvdb_mat4_row(m, 0);
    frustum[2] = pvdb_mat4_row(m, 3) + pvdb_mat4_row(m, 1);
    frustum[3] = pvdb_mat4_row(m, 3) - pvdb_mat4_row(m, 1);
    frustum[4] = pvdb_mat4_row(m, 3) + pvdb_mat4_row(m, 2);
    frustum[5] = pvdb_mat4_row(m, 3) - pvdb_mat4_row(m, 2);
}

//endregion

//region compute_region

PVDB_INLINE uint
pvdb_compute_region_size(
    PVDB_IN(ivec3)     size,
    PVDB_IN(ivec3)     local_size,
    PVDB_IN(ivec3)     global_size)
{
    const ivec3 size_global = size / local_size;
    return uint(size_global.x * size_global.y * size_global.z);
}


PVDB_INLINE ivec3
pvdb_compute_region(
    uint               index,
    PVDB_IN(ivec3)     size,
    PVDB_IN(ivec3)     local_size,
    PVDB_IN(ivec3)     global_size)
{
    const ivec3 size_global = size / local_size;
    return ivec3(
        int(index) / (size_global.y * size_global.z),
        (int(index) / size_global.z) % size_global.y,
        int(index) % size_global.z
    ) * local_size;
}

//endregion

//region ray


struct pvdb_ray {
    vec3 pos;
    vec3 dir;
};


PVDB_INLINE void
pvdb_ray_gen_primary(
    PVDB_OUT(pvdb_ray)  ray,
    mat4                proj,
    mat4                inv_mvp,
    vec2                uv)
{
    float fov = 2.0f * atan( 1.0f/proj[1][1] ) * 180.0f / PVDB_PI;
	float fx = tan(fov / 2.0f);

    vec4 near = inv_mvp * vec4(
        2.0f * fx * ( uv.x - 0.5f),
        2.0f * fx * ( uv.y - 0.5f),
        0.0f,
        1.0f
    );

    vec4 far = near + inv_mvp[2];
    near.xyz /= near.w;
    far.xyz /= far.w;
    ray.pos = near.xyz;
    ray.dir = normalize(far.xyz-near.xyz);
}


PVDB_INLINE bool
pvdb_ray_box_intersect(
    PVDB_IN(pvdb_ray)       ray,
    vec3                    bbox_min,
    vec3                    bbox_max,
    PVDB_INOUT(float)       tmin,
    PVDB_INOUT(float)       tmax,
    PVDB_INOUT(ivec3)       dir)
{
    vec3 dir_inv = vec3(1.0f) / ray.dir;
    vec3 t0 = (bbox_min - ray.pos) * dir_inv;
    vec3 t1 = (bbox_max - ray.pos) * dir_inv;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    float tnear = pvdb_max(tmin3);
    float tfar = pvdb_min(tmax3);
    bool hit = tnear <= tfar;
    tmin = max(tmin, tnear);
    tmax = min(tmax, tfar);
    dir = ivec3(int(tnear == tmin3.x), int(tnear == tmin3.y), int(tnear == tmin3.z));
    return hit;
}


PVDB_INLINE vec3
pvdb_ray_hit_normal(
    vec3                    dir,
    vec3                    hit,
    PVDB_IN(ivec3)          p_hit)
{
    // Compute the normal of the voxel [vmin, vmin+1] at the hit point
    // Note: This is not normalized when the ray hits an edge of the voxel exactly
    vec3 from_voxel_center = (hit - vec3(p_hit)) - 0.5f; // in [-1/2, 1/2]
    from_voxel_center -= dir * 0.01f; // Bias the sample point slightly towards the camera
    const vec3 center_abs = abs(from_voxel_center);
    const float max_coord = pvdb_max(center_abs);
    vec3 normal;
    normal.x = (center_abs.x == max_coord ? pvdb_copysign(1.0f, from_voxel_center.x) : 0.0f);
    normal.y = (center_abs.y == max_coord ? pvdb_copysign(1.0f, from_voxel_center.y) : 0.0f);
    normal.z = (center_abs.z == max_coord ? pvdb_copysign(1.0f, from_voxel_center.z) : 0.0f);
    return normal;
}

//endregion

//region dda

#define PVDB_DDA_FLT_MAX        3.402823466e+38
#define PVDB_DDA_FLT_MIN        0.001

#ifdef PVDB_C
struct pvdb_dda;
typedef PVDB_INOUT(pvdb_dda)           pvdb_dda_inout;
#else
#define pvdb_dda_inout                 PVDB_INOUT(pvdb_dda)
#endif


struct pvdb_dda {
    vec3    pos;            // origin of traversal
    vec3    dir;            // direction of traversal
    ivec3   dir_sign;       // integer mask indicating sign of dir in each coordinate direction
    vec3    t_delta;        // distance traversed in each coordinate direction per step
    vec3    t;              // dda time: x => t0, y => t1
    ivec3   p;              // position: node-local integer coord
    vec3    t_side;         // intersection time in each coordinate direction
    ivec3   mask;           // integer mask indicating which coordinate direction to step in next
};


PVDB_INLINE void
pvdb_dda_set_from_ray(
    pvdb_dda_inout          dda,
    PVDB_IN(pvdb_ray)       ray,
    PVDB_IN(vec3)           t)
{
    dda.pos = ray.pos;
    dda.dir = ray.dir;
    dda.dir_sign = ivec3(sign(ray.dir));
    dda.t = t;
}


PVDB_INLINE void
pvdb_dda_prepare_level(
    pvdb_dda_inout          dda,
    PVDB_IN(ivec3)          node_pos,
    vec3                    voxel_dim)
{
//    PVDB_PRINTF("DDA BEGIN LEVEL: dim=%f\n", voxel_dim.x);

    // time delta depends on voxel size at this level
    const vec3 delta_dir = voxel_dim / dda.dir;
    dda.t_delta = clamp(abs(delta_dir), -PVDB_DDA_FLT_MAX, PVDB_DDA_FLT_MAX);

    // get local position at this level
    const vec3 t0 = vec3(dda.t.x);
    const vec3 v_min = vec3(node_pos);
    const vec3 p = dda.pos + t0 * dda.dir;
    const vec3 p_local = (p - v_min) / voxel_dim; // @offset
    const vec3 p_local_floor = floor(p_local); // @offset
    dda.p = ivec3(p_local_floor); // @offset

    // advance time of each side
    const vec3 p_voxel_neg = p_local_floor - p_local + 0.5f; // @offset
    const vec3 p_voxel = p_voxel_neg * vec3(dda.dir_sign); // @offset
    dda.t_side = t0 + (p_voxel + 0.5f) * dda.t_delta;
}


PVDB_INLINE void
pvdb_dda_next(
    pvdb_dda_inout          dda)
{
    dda.mask.x = int(int(dda.t_side.x < dda.t_side.y) & int(dda.t_side.x <= dda.t_side.z));
    dda.mask.y = int(int(dda.t_side.y < dda.t_side.z) & int(dda.t_side.y <= dda.t_side.x));
    dda.mask.z = int(int(dda.t_side.z < dda.t_side.x) & int(dda.t_side.z <= dda.t_side.y));
    dda.t.y = (dda.mask.x != 0) ? dda.t_side.x : ((dda.mask.y != 0) ? dda.t_side.y : dda.t_side.z);
//    PVDB_PRINTF("DDA NEXT: t=%f\n", dda.t.y);
}


PVDB_INLINE void
pvdb_dda_step(
    pvdb_dda_inout          dda)
{
    dda.t.x = dda.t.y;
    dda.t_side += vec3(dda.mask) * dda.t_delta;
    dda.p += dda.mask * dda.dir_sign;
//    PVDB_PRINTF("DDA STEP: p=(%d, %d, %d)\n", dda.p.x, dda.p.y, dda.p.z);
}

//endregion

#endif //PVDB_MATH_H
