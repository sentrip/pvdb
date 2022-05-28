//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_DEBUGDRAWRTX_H
#define PVDB_DEBUGDRAWRTX_H

#include "../gpu/Pass.h"
#include <cstring>

namespace pvdb::gpu {

struct DebugDrawRTX : Subpass {
    VkRenderPass render_pass{};
    const Trees* trees{};
    Pipeline     pipeline{};

    void init_resources() final {}

    void init_pipelines() final {
        pipeline = ctx().create_graphics({{SRC_VERT, strlen(SRC_VERT)}, {}}, {{SRC_FRAG, strlen(SRC_FRAG)}, {}}, {render_pass, {}, true});
    }

    void record(gpu::Cmd cmd) final {
        cmd.bind_graphics(pipeline);
        (*trees)[0].push_const(ctx(), cmd);
        cmd.draw(3);
    }

    static constexpr const char* SRC_VERT = R"(#version 450
layout (location = 0) out vec2 outUV;
out gl_PerVertex { vec4 gl_Position; };

void main() {
	outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	gl_Position = vec4(outUV * 2.0f + -1.0f, 0.0f, 1.0f);
})";

    static constexpr const char* SRC_FRAG = R"(#version 450
#extension GL_EXT_debug_printf : enable
layout(location = 0) in vec2 inputUV;
layout(location = 0) out vec4 outputColor;

#define PVDB_TREE_READONLY
#include "pvdb/pvdb_tree.glsl"
#include "pvdb/pvdb_camera.glsl"
#include "pvdb/pvdb_tree_raycast.h"

layout(push_constant) uniform constants { pvdb_tree tree; } PC;

const vec4 colors [4] = {
    vec4(0.9 , 0.1, 0.1, 1.0),
    vec4(0.1, 0.9 , 0.1, 1.0),
    vec4(0.1, 0.1, 0.9 , 1.0),
    vec4(0.9 , 0.9 , 0.9 , 1.0)
};

void main() {
    if ((inputUV.x - 0.5f) >= 0.0f && (inputUV.x - 0.5f) <= 0.005f && (inputUV.y - 0.5f) >= 0.0f && (inputUV.y - 0.5f) <= 0.005f) {
        outputColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        return;
    }

    pvdb_ray ray;
    pvdb_ray_hit hit;
    pvdb_ray_gen_primary(ray, uCamera.proj, inverse(uCamera.mvp), vec2(inputUV.x, 1.0 - inputUV.y));
    if (!pvdb_raycast(PC.tree, ray, hit, ivec3(0)))
        discard;

    const float min_brightness = 0.1f;
    const float gamma_correction = 1.8f;
    const vec3 light_pos = vec3(-16.0, 1000.0, -128.0);

    vec3 hit_pos = ray.pos + (ray.dir * hit.t);
    vec4 voxel_color = colors[min(3, (hit.voxel & 0xffu) - 1)];
    vec3 out_color = voxel_color.rgb;
    out_color *= LightingPhong(light_pos - hit_pos, hit.normal, min_brightness);
    out_color = LightingGammaCorrection(out_color, gamma_correction);
    outputColor = vec4(out_color, voxel_color.a);
})";

};

}

#endif //PVDB_DEBUGDRAWRTX_H
