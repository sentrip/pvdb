//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_DEBUGFILL_H
#define PVDB_DEBUGFILL_H

#include "../gpu/Pass.h"
#include <cstring>

namespace pvdb::gpu {

struct DebugFill : Subpass {
    /// Public
    const Trees* trees{};

    void fill(u32 tree, const ivec3& offset, const ivec3& size, u32 value)
    {
        queue[count++] = {ivec4{offset, int(tree)}, ivec4{size, int(value)}};
    }

    /// Implementation
    struct FillData {
        ivec4 offset_tree{};
        ivec4 size_value{};
    };

    Pipeline                pipeline{};
    FillData                queue[64]{};
    u32                     count{};

    void init_resources() final {}

    void init_pipelines() final {
        ShaderMacro macro{"LOCAL_SIZE", to_string(LOCAL_SIZE.x)};
        pipeline = ctx().create_compute({{SRC, strlen(SRC)}, {&macro, 1}});
    }

    void record(gpu::Cmd cmd) final {
        assert(trees && "vdb::gpu::DebugFill: Did not set 'trees'");
        if (!count) return;

        cmd.bind_compute(pipeline);

        for (u32 i = 0; i < count; ++i) {
            const ivec3 global_size = ivec3(1) + (ivec3(queue[i].size_value) - ivec3(1)) / LOCAL_SIZE;
            push_const(queue[i]);
            auto& tree = (*trees)[uint(queue[i].offset_tree.w)];
            tree.push_const(ctx(), cmd, sizeof(FillData));
            cmd.dispatch(u32(global_size.x), u32(global_size.y), u32(global_size.z));
        }
        count = 0;
    }

    static constexpr ivec3 LOCAL_SIZE = ivec3(8);
    static constexpr const char* SRC = R"(#version 450
#include "pvdb/tree/pvdb_tree.glsl"

layout(push_constant) uniform constants {
	ivec3 offset;
    uint PAD;
    ivec3 size;
    uint value;
    pvdb_tree tree;
} PC;

layout(local_size_x = LOCAL_SIZE, local_size_y = LOCAL_SIZE, local_size_z = LOCAL_SIZE) in;
void main() {
    if (gl_GlobalInvocationID.x >= PC.size.x || gl_GlobalInvocationID.y >= PC.size.y || gl_GlobalInvocationID.z >= PC.size.z)
        return;
    ivec3 leaf_pos = ivec3(gl_GlobalInvocationID >> pvdb_log2dim(PC.tree, 0u));
    pvdb_tree_set(PC.tree, PC.offset + ivec3(gl_GlobalInvocationID), 1u + uint((leaf_pos.x & 1) + (leaf_pos.y & 1) + (leaf_pos.z & 1)));
})";
};


}

#endif //PVDB_DEBUGFILL_H
