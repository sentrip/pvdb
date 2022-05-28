//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_CVDB_CAMERA_H
#define PVDB_CVDB_CAMERA_H

#include "../fwd.h"
#include "../../pvdb/pvdb_math.h"

#include "../../gpu/Objects.h"

namespace pvdb {

struct UCamera {
    mat4  mvp{};
    mat4  proj{};
    vec4  frustum_planes[6]{};
    ivec4 offset{};
};


struct Camera {
    Camera() = default;
    void init(gpu_context ctx);
    void destroy();
    void update(const mat4& view, const mat4& proj) const;

private:
    u8 buffer_storage[GPU_BUFFER_SIZE_BYTES]{};
    u8 views_storage[2u * GPU_BUFFER_VIEW_SIZE_BYTES]{};
    const gpu::Context* p_ctx{};
};

}

/// FPSCamera
namespace pvdb {

struct FPSCamera {
    mat4  view{};
    mat4  proj{};
    vec3 position{}, forward{0.0f, 0.0f, 1.0f};
    float pitch{}, yaw{90};
    float fov{80.0f};
    float aspect_ratio{}, near_plane{0.01f}, far_plane{10000.0f};

    FPSCamera() { calculate_proj(); calculate_view(); }

    void set_position(const vec3& v)    { position = v; calculate_view(); }
    void look_at(const vec3& v)         { forward = normalize(v - position); calculate_pitch_yaw(); calculate_view(); }
    void rotate(float p, float y)       { pitch += p; yaw += y; pitch = max(-89.9999f, min(pitch, 89.9999f)); calculate_forward(); }
    void move(const vec3& v)            {
        position.y += v.y;
        position += vec3(v.x) * normalize(cross(forward, {0.0f, 1.0f, 0.0f}));
        position += vec3(v.z) * normalize(vec3(forward.x, 0.0f, forward.z));
        calculate_view();
    }

    void reset(float fv, float ar = 1.0f, float near = 0.01f, float far = 10000.0f) {
        fov = fv; aspect_ratio = ar; near_plane = near; far_plane = far;
        calculate_proj();
    }

private:

    void calculate_proj() {
        pvdb_mat4_projection_perspective(proj, pvdb_deg_to_rad(fov), aspect_ratio, near_plane, far_plane);
    }

    void calculate_view() {
        pvdb_mat4_look_at(view, position, position + forward, {0.0f, 1.0f, 0.0f});
        view = pvdb_mat4_inverted(view);
    }

    void calculate_forward() {
        const float r_pitch = pvdb_deg_to_rad(pitch);
        const float r_yaw = pvdb_deg_to_rad(yaw);
        forward = normalize(vec3{cos(r_pitch) * cos(r_yaw), sin(r_pitch), cos(r_pitch) * sin(r_yaw)});
        calculate_view();
    }

    void calculate_pitch_yaw() {
        yaw = pvdb_rad_to_deg(atan2(forward.z, forward.x));
        pitch = pvdb_rad_to_deg(asin(forward.y));
    }
};

}


#endif //PVDB_CVDB_CAMERA_H
