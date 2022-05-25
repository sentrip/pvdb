//
// Created by Djordje on 5/23/2022.
//

#include "Camera.h"
#include "../gpu/Context.h"
#include <cstring>

namespace pvdb {

void Camera::init(gpu_context ctx)
{
    p_ctx = &ctx;
    const u32 size = sizeof(UCamera) < 512 ? 512 : sizeof(UCamera);
    auto& buffer = *new (buffer_storage) gpu::Buffer;
    buffer = ctx.create_buffer(size * gpu::FRAMES_IN_FLIGHT, gpu::BufferType::UNIFORM, gpu::BufferUsage::CPU_TO_GPU);
    for (u32 i  = 0; i < gpu::FRAMES_IN_FLIGHT; ++i)
        new (views_storage + (i * GPU_BUFFER_VIEW_SIZE_BYTES)) gpu::BufferView{buffer, i * size, size};
    ctx.bind(PVDB_BINDING_CAMERA, gpu::BufferType::UNIFORM, {(const gpu::BufferView*)views_storage, gpu::FRAMES_IN_FLIGHT}, 0, gpu::Context::BIND_PER_FRAME);
}

void Camera::destroy()
{
    p_ctx->destroy_buffer(*((gpu::Buffer*)buffer_storage));
}

void Camera::update(const mat4& view, const mat4& proj) const
{
    UCamera data{};
    data.mvp = proj * view;
    data.proj = proj;
    pvdb_mat4_to_frustum(data.mvp, data.frustum_planes);
    memcpy(((const gpu::Buffer*)buffer_storage)->data + ((const gpu::BufferView*)views_storage)[p_ctx->frame].offset, &data, sizeof(UCamera));
}

}
