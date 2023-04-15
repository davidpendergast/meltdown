import numpy
import math
import random
import pygame
import typing
import os

import Box2D

import const as const
import src.utils as utils
import src.convexhull2 as convexhull
import src.sounds as sounds
import src.scenes as scenes
import src.gameplay as gameplay
from src.sprites import Spritesheet, UiSheet


class ParticleArray:

    def __init__(self, max_time_sec=const.PARTICLE_DURATION, cap=128):
        self.max_time_sec = max_time_sec
        self.start_idx = 0
        self.n = 0

        self.x = numpy.zeros((cap,), dtype=numpy.float16)
        self.y = numpy.zeros((cap,), dtype=numpy.float16)
        self.vx = numpy.zeros((cap,), dtype=numpy.float16)
        self.vy = numpy.zeros((cap,), dtype=numpy.float16)
        self.ax = numpy.zeros((cap,), dtype=numpy.float16)
        self.ay = numpy.zeros((cap,), dtype=numpy.float16)

        self.t = numpy.zeros((cap,), dtype=numpy.float16)
        self.energy = numpy.zeros((cap,), dtype=numpy.float16)

    def clear(self, new_cap=128):
        self.n = 0
        self.start_idx = 0
        if new_cap is not None:
            self._set_capacity(new_cap)

    def update(self, dt):
        self.vx += self.ax * dt
        self.vy += self.ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.t += dt

        self._clear_expired(self.max_time_sec)

    def all_active_indices(self):
        cap = self.get_capacity()
        for i in range(self.n):
            yield (self.start_idx + i) % cap

    def all_particle_xys(self, as_ints=False):
        if as_ints:
            for idx in self.all_active_indices():
                yield (int(self.x[idx]), int(self.y[idx]))
        else:
            for idx in self.all_active_indices():
                yield (self.x[idx], self.y[idx])

    def _clear_expired(self, max_time):
        cap = self.get_capacity()
        new_start_idx = self.start_idx
        new_n = self.n
        for i in range(0, self.n):
            idx = (self.start_idx + i) % cap
            if self.t[idx] < max_time:
                new_start_idx = idx
                break
            else:
                new_n -= 1
        self.n = new_n
        self.start_idx = new_start_idx

    def __len__(self):
        return self.n

    def get_capacity(self):
        return self.x.shape[0]

    def _set_capacity(self, cap):
        self.x = numpy.resize(self.x, (cap,))
        self.y = numpy.resize(self.y, (cap,))
        self.vx = numpy.resize(self.vx, (cap,))
        self.vy = numpy.resize(self.vy, (cap,))
        self.ax = numpy.resize(self.ax, (cap,))
        self.ay = numpy.resize(self.ay, (cap,))
        self.t = numpy.resize(self.t, (cap,))
        self.energy = numpy.resize(self.energy, (cap,))

    def add_particle(self, xy, velocity=None):
        cap = self.get_capacity()
        if self.n >= self.get_capacity():
            self._set_capacity(cap * 2)
            cap = self.get_capacity()

        if isinstance(velocity, (float, int)):
            rand = 2 * math.pi * random.random()
            v_xy = (velocity * math.cos(rand), velocity * math.sin(rand))
        else:
            v_xy = velocity

        idx = (self.start_idx + self.n) % cap
        self.x[idx] = xy[0]
        self.y[idx] = xy[1]
        self.vx[idx] = v_xy[0]
        self.vy[idx] = v_xy[1]
        self.ax[idx] = 0
        self.ay[idx] = 0
        self.t[idx] = 0
        self.energy[idx] = const.PARTICLE_ENERGY
        self.n += 1

    def __repr__(self):
        return f"{type(self).__name__}({len(self)} Particles)"


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    
    screen = utils.make_fancy_scaled_display(
        const.SCREEN_DIMS,
        scale_factor=2,
        outer_fill_color=(121, 0, 0),
        extra_flags=pygame.RESIZABLE)
    pygame.display.set_caption(const.GAME_TITLE)

    sounds.initialize(const.SOUND_DIR)
    gameplay.initialize_level_list(utils.res_path("levels/level_list.txt"))

    rad_surf = pygame.Surface(const.DIMS)

    clock = pygame.time.Clock()
    dt = 0

    # level = load_level_from_file("test.png")
    Spritesheet.load(utils.res_path("assets/sprites.png"))
    UiSheet.load(utils.res_path("assets"))

    scene_manager = scenes.SceneManager(scenes.MainMenuScene())

    frm_cnt = 0

    running = True
    while running and not scene_manager.should_quit:
        const.KEYS_PRESSED_THIS_FRAME.clear()
        const.KEYS_RELEASED_THIS_FRAME.clear()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                const.KEYS_PRESSED_THIS_FRAME.add(e.key)
                const.KEYS_HELD_THIS_FRAME.add(e.key)
            elif e.type == pygame.KEYUP:
                const.KEYS_RELEASED_THIS_FRAME.add(e.key)
                if e.key in const.KEYS_HELD_THIS_FRAME:
                    const.KEYS_HELD_THIS_FRAME.remove(e.key)

        scene_manager.update(dt)
        scene_manager.render(screen)

        pygame.display.flip()

        if frm_cnt % 15 == 14 and const.IS_DEV:
            caption_info = {'FPS': f"{clock.get_fps():.2f}"}
            for key, val in scene_manager.active_scene.get_caption_info().items():
                caption_info[key] = str(val)
            if len(caption_info) > 0:
                msg = ", ".join(f"{key}={val}" for (key, val) in caption_info.items())
                caption = f"{const.GAME_TITLE} ({msg})"
            else:
                caption = f"{const.GAME_TITLE}"
            pygame.display.set_caption(caption)

        dt = clock.tick(const.MAX_FPS) / 1000
        dt = min(dt, 1 / const.MIN_FPS)

        frm_cnt += 1

