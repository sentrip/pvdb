//
// Created by Djordje on 5/24/2022.
//

#ifndef PVDB_DEBUG_H
#define PVDB_DEBUG_H

#include "Camera.h"
#include "../gpu/DebugDevice.h"
#include <array>

namespace pvdb {

enum Control {
    MOVE_EAST,  MOVE_UP,   MOVE_NORTH,
    MOVE_WEST,  MOVE_DOWN, MOVE_SOUTH,
    LOCK_MOUSE,
    MAX
};

using Keybinds = std::array<Key, u32(Control::MAX)>;

static constexpr Keybinds DefaultKeybindings = [](){
    Keybinds b{};
    b[MOVE_NORTH] = KEY_W;
    b[MOVE_SOUTH] = KEY_S;
    b[MOVE_EAST] = KEY_D;
    b[MOVE_WEST] = KEY_A;
    b[MOVE_UP] = KEY_SPACE;
    b[MOVE_DOWN] = KEY_LSHIFT;
    b[LOCK_MOUSE] = KEY_BACKSPACE;
    return b;
}();

struct DebugController {
    float       speed{0.05f};
    float       sensitivity{0.05f};
    Keybinds    keys = DefaultKeybindings;
    FPSCamera   cam{};
    i32         dir[6]{};
    bool        mouse_locked{};

    void on_mouse_move(i32 move_x, i32 move_y) {
        cam.rotate(-float(move_y)*sensitivity, float(move_x)*sensitivity);
    }

    void on_key(gpu::DebugDevice& device, i32 k, bool pressed) {
        if (pressed && k == keys[LOCK_MOUSE]) { mouse_locked = !mouse_locked; device.lock_mouse(mouse_locked); return; }
        for (u32 i = 0; i < 6; ++i) {
            if (pressed)
                dir[i] |= int(k == keys[i]);
            else
                dir[i] &= ~int(k == keys[i]);
        }
    }

    void update(float dt) {
        cam.move((vec3(ivec3(dir[0], dir[1], dir[2])) - vec3(ivec3(dir[3], dir[4], dir[5]))) * speed * dt);
    }
};

}

#endif //PVDB_DEBUG_H
