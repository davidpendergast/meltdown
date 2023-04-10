import numpy
import math
import random
import pygame
import src.utils as utils

GAME_TITLE = "Containment"
LEVEL_DIR = "levels"

DIMS = (64, 48)
DISPLAY_SCALE_FACTOR = 4
EXTRA_SCREEN_HEIGHT = 48
SCREEN_DIMS = (DISPLAY_SCALE_FACTOR * DIMS[0],
               DISPLAY_SCALE_FACTOR * DIMS[1] + EXTRA_SCREEN_HEIGHT)

KEYS_HELD_THIS_FRAME = None

AMBIENT_ENERGY_DECAY_RATE = 0.2

SPAWN_RATE = 100
PARTICLE_DURATION = 5

PARTICLE_VELOCITY = 10
PARTICLE_ENERGY = 400
ENERGY_TRANSFER_ON_COLLISION = 0.1

MOVE_RESOLUTION = 4
MOVE_SPEED = 16

MAX_DOSE = PARTICLE_ENERGY * 4

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

class Spritesheet:
    player = []
    barrels = []

    heart_icon = None
    skull_icon = None
    bar_empty = None
    bar_full = None

    @staticmethod
    def load(filepath):
        def scale(surf, factor):
            return pygame.transform.scale_by(surf, (factor, factor))
        img = pygame.image.load(filepath).convert_alpha()
        x = 0
        y = 0
        Spritesheet.player = img.subsurface([x, y, 32, 32])
        Spritesheet.barrels = [img.subsurface([(x := x + 32), y, 16, 32]),
                               img.subsurface([(x := x + 16), y, 16, 32])]
        y += 32
        bar_sc = 2
        Spritesheet.heart_icon = scale(img.subsurface([0, y, 17, 17]), bar_sc)
        Spritesheet.skull_icon = scale(img.subsurface([17 + 46, y, 17, 17]), bar_sc)
        Spritesheet.bar_empty = scale(img.subsurface([17, y, 46, 17]), bar_sc)
        Spritesheet.bar_full = scale(img.subsurface([15, y + 16, 50, 17]), bar_sc)


class Entity:

    def __init__(self, xy, dims=(1, 1)):
        self.xy = xy
        self.dims = dims

    def get_rect(self):
        return (self.xy[0], self.xy[1], self.dims[0], self.dims[1])

    def update(self, dt, level, **kwargs):
        pass

    def render(self, surf, color):
        rect = self.get_rect()
        pygame.draw.rect(surf, color, (int(rect[0] * DISPLAY_SCALE_FACTOR),
                                       int(rect[1] * DISPLAY_SCALE_FACTOR),
                                       max(1, round(rect[2] * DISPLAY_SCALE_FACTOR)),
                                       max(1, round(rect[3] * DISPLAY_SCALE_FACTOR))))

    def resolve_rect_collision_with_level(self, level, geom_thresh=0.1) -> bool:
        rect = self.get_rect()
        if level.is_colliding(rect, geom_thresh=geom_thresh):
            x, y = rect[0], rect[1]
            x_to_check = (x, int(x) + 1, int(x + rect[2]) - rect[2])
            y_to_check = (y, int(y) + 1, int(y + rect[3]) - rect[3])

            candidate_xys = []
            for cx in x_to_check:
                for cy in y_to_check:
                    candidate_xys.append((cx, cy))
            candidate_xys.sort(key=lambda cxy_: (x - cxy_[0])**2 + (y - cxy_[1])**2)

            for cxy in candidate_xys:
                if not level.is_colliding((cxy[0], cxy[1], rect[2], rect[3])):
                    self.xy[0] = cxy[0]
                    self.xy[1] = cxy[1]
                    return True
            print(f"Failed to resolve collision at: {self.xy}")
            return False
        return True


class ParticleEmitter(Entity):

    def __init__(self, xy, dims=(1, 1), weight=1):
        super().__init__(xy, dims=dims)
        self.weight = weight
        self.accum_t = 0

    def update(self, dt, level, rate=1, **kwargs):
        super().update(dt, level)
        self.accum_t += dt
        n_to_spawn = int(rate * self.accum_t)
        self.accum_t -= n_to_spawn / rate

        for _ in range(n_to_spawn):
            self.spawn_particle(level)

    def spawn_particle(self, level: 'Level'):
        level.particles.add_particle(self.xy, PARTICLE_VELOCITY)


class ParticleAbsorber(Entity):

    def __init__(self, xy, absorb_rate=1, **kwargs):
        super().__init__(xy, **kwargs)
        self.absorb_rate = absorb_rate
        self.energy_accum = 0
        self.decay_rate = AMBIENT_ENERGY_DECAY_RATE
        self.update_particles = True

    def absorb(self, dt, level, p_idx):
        to_absorb = level.particles.energy[p_idx] * min(1, self.absorb_rate * dt)
        self.energy_accum += to_absorb
        if self.update_particles:
            level.particles.energy[p_idx] -= to_absorb

    def update(self, dt, level, **kwargs):
        super().update(dt, level, **kwargs)
        self.energy_accum *= (1 - AMBIENT_ENERGY_DECAY_RATE * dt)


class Player(ParticleAbsorber):

    def __init__(self, xy, dims=(2, 2)):
        super().__init__(xy, dims=dims)

    def update(self, dt, level, **kwargs):
        super().update(dt, level)
        move_dir = pygame.Vector2()

        keys = KEYS_HELD_THIS_FRAME
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            move_dir.x -= 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            move_dir.x += 1
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_dir.y -= 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_dir.y += 1
        if move_dir.magnitude() > 0:
            move_dir.scale_to_length(MOVE_SPEED / MOVE_RESOLUTION)
            for _ in range(MOVE_RESOLUTION):
                self.xy += move_dir * dt
                self.resolve_rect_collision_with_level(level)
        else:
            self.resolve_rect_collision_with_level(level)

    def render(self, surf, color):
        rect = self.get_rect()
        center_xy_on_screen = (
            DISPLAY_SCALE_FACTOR * (rect[0] + rect[2] / 2),
            DISPLAY_SCALE_FACTOR * (rect[1] + rect[3] / 2))
        sprite = Spritesheet.player
        blit_xy = (center_xy_on_screen[0] - sprite.get_width() // 2,
                   center_xy_on_screen[1] - sprite.get_height() // 2)
        surf.blit(Spritesheet.player, blit_xy)


class Level:

    def __init__(self, size, spawn_rate=SPAWN_RATE):
        self.size = size
        self.particles: ParticleArray = ParticleArray()
        self.spawn_rate = spawn_rate
        self.spatial_hash = {}

        self.player = None
        self.absorbers = []
        self.emitters = []

        self.geometry = pygame.Surface(size).convert()
        self.energy = numpy.zeros(size, dtype=numpy.float16)

    def update(self, dt):
        self.update_emitters(dt)
        self.update_particles(dt)
        if self.player is not None:
            self.player.update(dt, self)
        self.update_absorbers(dt)

        self.energy *= (1 - AMBIENT_ENERGY_DECAY_RATE * dt)

    def update_emitters(self, dt):
        total_weight = sum((emit.weight for emit in self.emitters), start=0)
        for emitter in self.emitters:
            rate = self.spawn_rate * emitter.weight / total_weight
            emitter.update(dt, self, rate=rate)

    def update_particles(self, dt):
        self.particles.update(dt)
        self._handle_collisions(dt)

    def update_absorbers(self, dt):
        for absorber in self.absorbers:
            for idx in self.all_particles_indices_in_rect(absorber.get_rect()):
                absorber.absorb(dt, self, idx)

    def add_entity(self, ent):
        if isinstance(ent, ParticleEmitter):
            self.emitters.append(ent)
        if isinstance(ent, ParticleAbsorber):
            self.absorbers.append(ent)
        if isinstance(ent, Player):
            if self.player is not None:
                raise ValueError("level already has a player")
            self.player = ent

    def remove_entity(self, ent):
        if isinstance(ent, ParticleEmitter):
            self.emitters.remove(ent)
        if isinstance(ent, ParticleAbsorber):
            self.absorbers.remove(ent)
        if isinstance(ent, Player):
            self.player = None  # hope it's the same player

    def _handle_collisions(self, dt):
        self.spatial_hash.clear()
        for idx in self.particles.all_active_indices():
            x = int(self.particles.x[idx])
            y = int(self.particles.y[idx])
            if self.is_valid((x, y)):
                geom = self.get_geom_at((x, y))
                if geom > 0 and random.random() < dt * 1000:
                    energy_lost = ENERGY_TRANSFER_ON_COLLISION * self.particles.energy[idx]
                    self.energy[x][y] += energy_lost
                    self.particles.energy[idx] -= energy_lost

                    new_angle = random.random() * 2 * math.pi
                    self.particles.vx[idx] = PARTICLE_VELOCITY * math.cos(new_angle)
                    self.particles.vy[idx] = PARTICLE_VELOCITY * math.sin(new_angle)
                if (x, y) not in self.spatial_hash:
                    self.spatial_hash[(x, y)] = []
                self.spatial_hash[(x, y)].append(idx)

    def all_particles_indices_in_rect(self, rect, thresh=0.001):
        for xy in self.all_cells_in_rect(rect, thresh=thresh):
            if xy in self.spatial_hash:
                for idx in self.spatial_hash[xy]:
                    yield idx

    def render_geometry(self, surf: pygame.Surface):
        surf.fill((255, 255, 255))
        surf.blit(self.geometry, (0, 0), special_flags=pygame.BLEND_RGB_SUB)
        # surf.blit(self.geometry, (0, 0))

    def is_valid(self, xy):
        return 0 <= xy[0] < self.size[0] and 0 <= xy[1] < self.size[1]

    def get_geom_at(self, xy, or_else=1):
        if not self.is_valid(xy):
            return or_else
        else:
            return (255 - self.geometry.get_at(xy)[0]) / 255

    def all_cells_in_rect(self, rect, thresh=0.001):
        for x in range(int(rect[0] + thresh), int(rect[0] + rect[2] + 1 - thresh)):
            for y in range(int(rect[1] + thresh), int(rect[1] + rect[3] + 1 - thresh)):
                yield (x, y)

    def is_colliding(self, rect, thresh=0.001, geom_thresh=0.1):
        for xy in self.all_cells_in_rect(rect, thresh=thresh):
                if self.get_geom_at(xy) >= geom_thresh:
                    return True
        return False

    def render_particles(self, surf: pygame.Surface):
        for xy in self.particles.all_particle_xys(as_ints=True):
            if self.is_valid(xy):
                surf.set_at(xy, "red")

    def render_energy(self, surf: pygame.Surface):
        rbuf = pygame.surfarray.pixels_red(surf)
        max_energy = PARTICLE_ENERGY * 1.5  # numpy.max(self.energy)
        if max_energy > 0:
            rbuf[:] = (numpy.clip(self.energy / max_energy, 0, 1) * 255).astype(numpy.int16)

    def render_entities(self, surf: pygame.Surface):
        if self.player is not None:
            self.player.render(surf, "blue")

def load_level_from_file(filename) -> Level:
    img = pygame.image.load(f'{LEVEL_DIR}/{filename}').convert()
    size = img.get_size()
    res = Level(size)

    for y in range(size[1]):
        for x in range(size[0]):
            clr = img.get_at((x, y))
            if clr == (255, 0, 0):
                res.add_entity(ParticleEmitter((x + 0.5, y + 0.5)))
                img.set_at((x, y), (255, 255, 255))
            elif clr == (0, 0, 255):
                res.add_entity(Player((x + 1, y + 1)))
                img.set_at((x, y), (255, 255, 255))

    res.geometry.blit(img, (0, 0))
    return res


def render_dose_bar(surf: pygame.Surface, rect, pcnt):
    # pygame.draw.rect(surf, (0, 255, 0), rect, width=0)

    icon_w = Spritesheet.heart_icon.get_width()
    icon_h = Spritesheet.heart_icon.get_height()
    pcnt = min(1, max(0, pcnt))

    bar_rect = (rect[0] + icon_w, rect[1], rect[2] - icon_w * 2, rect[3])
    filled_bar_rect = (bar_rect[0] - 2 * int(icon_w / 17), rect[1], bar_rect[2] + 3 * int(icon_w / 17), rect[3])

    xformed_full_bar_sprite = pygame.transform.scale(Spritesheet.bar_full, filled_bar_rect[2:4])
    full_bar_sprite = xformed_full_bar_sprite.subsurface((0, 0, int(filled_bar_rect[2] * pcnt), icon_h))
    surf.blit(full_bar_sprite, filled_bar_rect)

    surf.blit(Spritesheet.heart_icon, rect[0:2])
    surf.blit(Spritesheet.skull_icon, (rect[0] + rect[2] - icon_w, rect[1]))

    surf.blit(pygame.transform.scale(Spritesheet.bar_empty, bar_rect[2:4]),
              (rect[0] + icon_w, rect[1], rect[2] - icon_w * 2, rect[3]))


if __name__ == "__main__":
    screen = utils.make_fancy_scaled_display(SCREEN_DIMS, scale_factor=2, extra_flags=pygame.RESIZABLE)
    pygame.display.set_caption(GAME_TITLE)

    rad_surf = pygame.Surface(DIMS)

    clock = pygame.time.Clock()
    dt = 0

    level = load_level_from_file("test.png")
    Spritesheet.load("assets/sprites.png")

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

        KEYS_HELD_THIS_FRAME = pygame.key.get_pressed()

        level.update(dt)

        screen.fill("black")
        rad_surf.fill("black")

        if KEYS_HELD_THIS_FRAME[pygame.K_p]:
            level.render_geometry(rad_surf)
            level.render_particles(rad_surf)
        else:
            level.render_energy(rad_surf)

        screen.blit(pygame.transform.scale_by(rad_surf, (DISPLAY_SCALE_FACTOR,) * 2), (0, 0))
        level.render_entities(screen)

        dose_bar_size = (2 * SCREEN_DIMS[0] // 3, Spritesheet.heart_icon.get_height())
        render_dose_bar(screen, (SCREEN_DIMS[0] // 2 - dose_bar_size[0] // 2,
                                 SCREEN_DIMS[1] - EXTRA_SCREEN_HEIGHT // 2 - dose_bar_size[1] // 2,
                                 dose_bar_size[0], dose_bar_size[1]),
                        level.player.energy_accum / MAX_DOSE)

        pygame.display.flip()

        if frm_cnt % 15 == 14:
            pygame.display.set_caption(f"{GAME_TITLE} (FPS={clock.get_fps():.2f}, "
                                       f"N={len(level.particles)}, pE={level.player.energy_accum})")

        dt = clock.tick() / 1000
        frm_cnt += 1



