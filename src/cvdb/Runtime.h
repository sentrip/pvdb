//
// Created by Djordje on 5/23/2022.
//

#ifndef PVDB_CVDB_RUNTIME_H
#define PVDB_CVDB_RUNTIME_H

#include "fwd.h"
#include "../pvdb/pvdb_config.h"

namespace pvdb {

struct RuntimeDesc {
    void*   device{};
    void*   vma{};
    void*   queue_compute{};
    void*   queue_graphics{};
    u32     queue_family{};
    void*   debug_draw_render_pass{};
};


struct Runtime {
    void                init(const RuntimeDesc& desc);
    void                destroy();

    gpu::Context&       context();
    const gpu::Context& context() const;
    const gpu::Submit&  last_submit() const;

    Camera&             camera();
    const Camera&       camera() const;

    Trees&              trees();
    const Trees&        trees() const;

    enum class Draw     { RTX, MESH, DEFAULT };
    void                draw(void* vk_command_buffer, Draw d = Draw::DEFAULT) const;

    void                debug_fill(u32 tree, const ivec3& offset, const ivec3& size, u32 value) const;

private:
    struct Storage;
    Storage* self{};
};

}

#endif //PVDB_CVDB_RUNTIME_H
