//
// Created by Djordje on 5/23/2022.
//

#include "Runtime.h"
#include "Tree.h"
#include "Camera.h"

#include "../gpu/Context.h"
#include "passes/DebugFill.h"
#include "passes/DebugDrawRTX.h"

namespace pvdb {

struct DebugDraw {
    VkRenderPass        render_pass{};
    gpu::DebugDrawRTX   subpass_rtx{};
    gpu::DebugDrawRTX   subpass_mesh{};
    gpu::Pass           pass{};
};


struct Passes {
    gpu::Pass voxels{};

    struct {
        gpu::DebugFill debug_fill{};
    } sub;
};


struct Runtime::Storage {
    gpu::Context    context{};
    Camera          camera{};
    Trees           trees{};
    Passes          passes{};
    DebugDraw       debug_draw{};
};


gpu::Context&           Runtime::context()            { assert(self); return self->context; }
const gpu::Context&     Runtime::context()      const { assert(self); return self->context; }
const gpu::Submit&      Runtime::last_submit()  const { assert(self); return self->passes.voxels.submit(); }
Camera&                 Runtime::camera()             { assert(self); return self->camera; }
const Camera&           Runtime::camera()       const { assert(self); return self->camera; }
Trees&                  Runtime::trees()              { assert(self); return self->trees; }
const Trees&            Runtime::trees()        const { assert(self); return self->trees; }


void Runtime::init(const RuntimeDesc& desc)
{
    self = new Storage;
    self->debug_draw.render_pass = VkRenderPass(desc.debug_draw_render_pass);
    self->context.init({VkDevice(desc.device), VmaAllocator(desc.vma), VkQueue(desc.queue_compute), VkQueue(desc.queue_graphics), desc.queue_family});
    self->camera.init(self->context);
    self->trees.init(self->context);

    self->context.add_pass(self->passes.voxels, gpu::IsGraphics::NO);
    self->passes.sub.debug_fill.trees = &self->trees;
    self->passes.voxels.add(self->passes.sub.debug_fill);

    self->context.setup();

    if (self->debug_draw.render_pass) {
        gpu::ResourceSync sync{}; // The subpasses will never call Subpass::read() or Subpass::write() so we can use a pointer to a temporary
        self->debug_draw.pass.init(self->context, sync, gpu::IsGraphics::YES);

        self->debug_draw.subpass_rtx.trees = &self->trees;
        self->debug_draw.subpass_rtx.render_pass = self->debug_draw.render_pass;
        self->debug_draw.pass.add(self->debug_draw.subpass_rtx);

        self->debug_draw.pass.setup();
    }
}

void Runtime::destroy()
{
    if (!self) return;

    self->context.wait();

    if (self->debug_draw.render_pass)
        self->debug_draw.pass.destroy();

    self->camera.destroy();
    self->context.destroy();

    delete self;
    self = nullptr;
}

void Runtime::draw(void *vk_command_buffer, Draw d) const
{
    const bool uses_meshes = false;
    assert((d != Draw::MESH || uses_meshes) && "Cannot draw meshes when meshes are disabled");
    gpu::Subpass& default_subpass  = uses_meshes ? (gpu::Subpass&)self->debug_draw.subpass_mesh : (gpu::Subpass&)self->debug_draw.subpass_rtx;
    gpu::Subpass& explicit_subpass = d == Draw::MESH ? (gpu::Subpass&)self->debug_draw.subpass_mesh : (gpu::Subpass&)self->debug_draw.subpass_rtx;
    gpu::Subpass& subpass = d == Draw::DEFAULT ? default_subpass : explicit_subpass;
    subpass.record(gpu::Cmd{VkCommandBuffer(vk_command_buffer)});
}

void Runtime::debug_fill(u32 tree, const ivec3& offset, const ivec3& size, u32 value) const
{
    assert(self);
    self->passes.sub.debug_fill.fill(tree, offset, size, value);
}

}
