//
// Created by Djordje on 5/15/2022.
//

#ifndef VDB_GPU_PASS_H
#define VDB_GPU_PASS_H

#include "Objects.h"

namespace pvdb::gpu {


struct Subpass {
    Subpass() = default;

    /// Stage 1: initialize buffers, images and bind
    virtual void init_resources() = 0;

    /// Stage 2: initialize pipelines
    virtual void init_pipelines() = 0;

    /// Stage 3: setup buffers using commands (optional)
    virtual void setup_resources(Cmd cmd) {}

    /// Stage 4: record commands
    virtual void record(Cmd cmd) = 0;

    /// Stage 5: destroy
    virtual void destroy() {}

    /// Context for resource management
    const Context& ctx() const { return *_ctx; }

    /// Record API helpers
    void        read(const BufferView& buffer, PipelineStage stage = PipelineStage::COMPUTE) const;
    void        write(const BufferView& buffer, PipelineStage stage = PipelineStage::COMPUTE) const;

    void        read(const Image& image, PipelineStage stage = PipelineStage::COMPUTE) const;
    void        write(const Image& image, PipelineStage stage = PipelineStage::COMPUTE) const;

    template<typename T>
    void        push_const(const T& value, u32 offset = 0) const { push_const(&value, sizeof(T), offset); }

private:
    void        push_const(const void* data, u32 size, u32 offset) const;

    friend struct Pass;
    const Context*  _ctx{};
    Pass*           pass{};
};


struct Pass {
    Pass() = default;

    void init(const Context& ctx, ResourceSync& sync, IsGraphics g = IsGraphics::NO);
    void destroy();

    void add(Subpass& subpass);
    void setup();
    void execute();
    void wait();
    u64  exec_time(u32 subpass = -1u) const;

    /// Subpass-Subpass sync
    ResourceSync& sync() const { return *_sync; }

    /// Pass-Pass sync
    const Submit& submit() const { return _submit; }
    void waits_for(const Pass& pass, PipelineStage stage);
    void set_last(bool last) { is_last = last; }

private:
    Submit          _submit{};
    ResourceSync*   _sync{};
    Subpass*        subpasses[32]{};
    u64             subpass_execution_time[33]{};
    u32             subpass_count{};
    Timer           timer{};
    IsGraphics      is_graphics{};
    const Context*  context{};
    bool            is_setup{};
    bool            is_last{};
};

}


#endif //VDB_GPU_PASS_H
