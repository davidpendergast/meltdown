import numpy
import math
import random
import pygame
import src.utils as utils

GAME_TITLE = "Containment"
LEVEL_DIR = "levels"

AMBIENT_ENERGY_DECAY_RATE = 0.1

SPAWN_RATE = 50
PARTICLE_DURATION = 5

PARTICLE_VELOCITY = 10
PARTICLE_ENERGY = 100
ENERGY_TRANSFER_ON_COLLISION = 0.1

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
        self.energy = numpy.zeros((cap,), dtype=numpy.float16)

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
        self.energy[idx] = PARTICLE_ENERGY
        self.n += 1

    def __repr__(self):
        return f"{type(self).__name__}({len(self)} Particles)"


class ParticleEmitter:

    def __init__(self, xy, weight=1):
        self.xy = xy
        self.weight = weight

        self.accum_t = 0

    def update(self, dt, rate, level):
        self.accum_t += dt
        n_to_spawn = int(rate * self.accum_t)
        self.accum_t -= n_to_spawn / rate

        for _ in range(n_to_spawn):
            self.spawn_particle(level)

    def spawn_particle(self, level: 'Level'):
        level.particles.add_particle(self.xy, PARTICLE_VELOCITY)


class Level:

    def __init__(self, size, spawn_rate=SPAWN_RATE):
        self.size = size
        self.particles: ParticleArray = ParticleArray()
        self.spawn_rate = spawn_rate

        self.absorbers = []
        self.emitters = []

        self.geometry = pygame.Surface(size).convert()
        self.energy = numpy.zeros(size, dtype=numpy.float16)

    def update(self, dt):
        self.update_emitters(dt)
        self.update_particles(dt)

        self.energy *= (1 - AMBIENT_ENERGY_DECAY_RATE * dt)

    def update_emitters(self, dt):
        total_weight = sum((emit.weight for emit in self.emitters), start=0)
        for emitter in self.emitters:
            rate = self.spawn_rate * emitter.weight / total_weight
            emitter.update(dt, rate, self)

    def update_particles(self, dt):
        self.particles.update(dt)
        self._handle_collisions(dt)

    def _handle_collisions(self, dt):
        for idx in self.particles.all_active_indices():
            x = int(self.particles.x[idx])
            y = int(self.particles.y[idx])
            if self.is_valid((x, y)):
                geom = (255 - self.geometry.get_at((x, y))[0]) / 255
                if geom > 0 and random.random() < dt * 1000:
                    energy_lost = ENERGY_TRANSFER_ON_COLLISION * self.particles.energy[idx]
                    self.energy[x][y] += energy_lost
                    self.particles.energy[idx] -= energy_lost

                    new_angle = random.random() * 2 * math.pi
                    self.particles.vx[idx] = PARTICLE_VELOCITY * math.cos(new_angle)
                    self.particles.vy[idx] = PARTICLE_VELOCITY * math.sin(new_angle)

    def render_geometry(self, surf: pygame.Surface):
        #surf.fill((255, 255, 255))
        #surf.blit(self.geometry, (0, 0), special_flags=pygame.BLEND_RGB_SUB)
        surf.blit(self.geometry, (0, 0))

    def is_valid(self, xy):
        return 0 <= xy[0] < self.size[0] and 0 <= xy[1] < self.size[1]

    def render_particles(self, surf: pygame.Surface):
        for xy in self.particles.all_particle_xys(as_ints=True):
            if self.is_valid(xy):
                surf.set_at(xy, "red")

    def render_energy(self, surf: pygame.Surface):
        rbuf = pygame.surfarray.pixels_red(surf)
        max_energy = numpy.max(self.energy)
        if max_energy > 0:
            rbuf[:] = ((self.energy / max_energy) * 255).astype(numpy.int16)

def load_level_from_file(filename) -> Level:
    img = pygame.image.load(f'{LEVEL_DIR}/{filename}').convert()
    size = img.get_size()
    res = Level(size)

    for y in range(size[1]):
        for x in range(size[0]):
            clr = img.get_at((x, y))
            if clr == (255, 0, 0):
                res.emitters.append(ParticleEmitter((x, y)))
                img.set_at((x, y), (255, 255, 255))
            elif clr == (0, 0, 255):
                # TODO spawn player
                img.set_at((x, y), (255, 255, 255))

    res.geometry.blit(img, (0, 0))
    return res


if __name__ == "__main__":
    DIMS = (64, 48)
    screen = utils.make_fancy_scaled_display(DIMS, scale_factor=8)
    pygame.display.set_caption(GAME_TITLE)

    clock = pygame.time.Clock()
    dt = 0

    level = load_level_from_file("test.png")

    frm_cnt = 0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or \
                    (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    level = load_level_from_file("test.png")

        keys = pygame.key.get_pressed()

        level.update(dt)

        screen.fill("black")
        if keys[pygame.K_p]:
            level.render_geometry(screen)
            level.render_particles(screen)
        else:
            level.render_energy(screen)

        pygame.display.flip()

        if frm_cnt % 15 == 14:
            pygame.display.set_caption(f"{GAME_TITLE} (FPS={clock.get_fps():.2f}, N={len(level.particles)})")

        dt = clock.tick() / 1000
        frm_cnt += 1



