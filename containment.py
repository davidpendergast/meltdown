import numpy
import math
import random
import pygame
import src.utils as utils

GAME_TITLE = "Containment"
PARTICLE_VELOCITY = 100
DECAY_RATE = 0.1
PARTICLE_DURATION = 1

class ParticleArray:

    def __init__(self, max_time_sec=PARTICLE_DURATION, cap=128):
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

    def clear(self, new_cap=128):
        self.n = 0
        self.start_idx = 0
        if new_cap is not None:
            self._set_capacity(new_cap)

    def update(self, dt):
        # TODO real accel
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
        self.n += 1

    def __repr__(self):
        return f"{type(self).__name__}({len(self)} Particles)"


class ParticleEmitter:

    def __init__(self, xy, spawnrate):
        self.xy = xy
        self.spawnrate = spawnrate

        self.accum_t = 0

    def update(self, dt, level):
        self.accum_t += dt
        n_to_spawn = int(self.spawnrate * self.accum_t)
        self.accum_t -= n_to_spawn / self.spawnrate

        for _ in range(n_to_spawn):
            self.spawn_particle(level)

    def spawn_particle(self, level: 'Level'):
        level.particles.add_particle(self.xy, PARTICLE_VELOCITY)


class Level:

    def __init__(self, size):
        self.size = size
        self.particles: ParticleArray = ParticleArray()
        self.absorbers = []
        self.emitters = []

    def update(self, dt):
        self.update_emitters(dt)
        self.update_particles(dt)

    def update_emitters(self, dt):
        for emitter in self.emitters:
            emitter.update(dt, self)

    def update_particles(self, dt):
        self.particles.update(dt)

    def render_particles(self, surf: pygame.Surface):
        for xy in self.particles.all_particle_xys(as_ints=True):
            surf.set_at(xy, "red")


if __name__ == "__main__":
    DIMS = (240, 120)
    screen = utils.make_fancy_scaled_display(DIMS, scale_factor=4)
    pygame.display.set_caption(GAME_TITLE)

    clock = pygame.time.Clock()
    dt = 0

    level = Level(DIMS)
    level.emitters.append(ParticleEmitter((DIMS[0] // 2, DIMS[1] // 2), 250))

    frm_cnt = 0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or \
                    (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False

        level.update(dt)

        screen.fill("black")
        level.render_particles(screen)

        pygame.display.flip()

        if frm_cnt % 15 == 14:
            pygame.display.set_caption(f"{GAME_TITLE} (FPS={clock.get_fps():.2f}, N={len(level.particles)})")

        dt = clock.tick() / 1000
        frm_cnt += 1



