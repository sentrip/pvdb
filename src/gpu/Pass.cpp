//
// Created by Djordje on 5/15/2022.
//

#include "Pass.h"
#include "Context.h"
#include "tinystd_assert.h"

namespace pvdb::gpu {

void Subpass::push_const(const void* data, u32 size, u32 offset)    const { ctx().push_const(pass->submit().cmd(ctx().frame), data, size, offset); }
void Subpass::read(const BufferView& buffer, PipelineStage stage)   const { pass->sync().read(pass->submit().cmd(ctx().frame), buffer, stage); }
void Subpass::write(const BufferView& buffer, PipelineStage stage)  const { pass->sync().write(pass->submit().cmd(ctx().frame), buffer, stage); }
void Subpass::read(const Image& image, PipelineStage stage)         const { pass->sync().read(pass->submit().cmd(ctx().frame), image, stage); }
void Subpass::write(const Image& image, PipelineStage stage)        const { pass->sync().write(pass->submit().cmd(ctx().frame), image, stage); }
void Pass::waits_for(const Pass& pass, PipelineStage stage)               { _submit.waits_for(pass.submit(), stage); }

void Pass::init(const Context& ctx, ResourceSync& s, IsGraphics g)
{
    context = &ctx;
    _sync = &s;
    is_graphics = g;
    _submit = context->create_submit(is_graphics);
}

void Pass::destroy()
{
    if (context == nullptr)
        return;
    for (u32 i = 0; i < subpass_count; ++i)
        subpasses[i]->destroy();
    timer.destroy(context->device, context->vma);
    subpass_count = 0;
    _submit.destroy();
    is_setup = false;
}

void Pass::add(Subpass& subpass)
{
    tassert(context && "Must call Pass::init() before calling Pass::add()");
    subpass._ctx = context;
    subpass.pass = this;
    subpass.init_resources();
    subpasses[subpass_count++] = &subpass;
}

void Pass::setup()
{
    tassert(context && "Must call Pass::init() before calling Pass::setup()");

    if (subpass_count == 0)
        return;

    if (!is_setup) {
        tassert(context->pipeline_layout && "Must call Context::update() at least once before calling Pass::setup()");

        for (u32 i = 0; i < subpass_count; ++i)
            subpasses[i]->init_pipelines();

        timer = Timer::create(context->device, context->vma, (subpass_count + 1u) * 2);

        auto cmd = context->begin_setup();
        timer.reset(cmd);
        timer.record(cmd, is_graphics);
        for (u32 i = 0; i < subpass_count; ++i) {
            timer.record(cmd, is_graphics);
            subpasses[i]->setup_resources(cmd);
            timer.record(cmd, is_graphics);
        }
        timer.record(cmd, is_graphics);
        timer.copy_results(cmd);
        context->end_setup();
        context->wait_setup();
        is_setup = true;
    }
}

void Pass::execute()
{
    tassert(context && "Must call Pass::init() before calling Pass::execute()");

    setup();
    wait();

    auto& ctx = *context;
    auto cmd = _submit.begin(ctx.frame);
    ctx.bind(cmd, is_graphics);
    timer.reset(cmd);
    timer.record(cmd, is_graphics);
    for (u32 i = 0; i < subpass_count; ++i) {
        timer.record(cmd, is_graphics);
        subpasses[i]->record(cmd);
        timer.record(cmd, is_graphics);
    }
    timer.record(cmd, is_graphics);
    timer.copy_results(cmd);
    _submit.submit(ctx, ctx.frame, is_last);

    if (ctx.frame == FRAMES_IN_FLIGHT - 1u)
        _sync->clear();
}

void Pass::wait()
{
    tassert(context && "Must call Pass::init() before calling Pass::wait()");
    _submit.wait(context->frame);
    subpass_execution_time[subpass_count] = timer.delta_time(0, (2u * subpass_count) - 1u);
    for (u32 i = 0; i < subpass_count; ++i)
        subpass_execution_time[i] = timer.delta_time(1u + (i * 2u), 2u + (i * 2u));
}

u64 Pass::exec_time(u32 subpass) const
{
    return subpass_execution_time[subpass == -1u ? subpass_count : subpass];
}

}
