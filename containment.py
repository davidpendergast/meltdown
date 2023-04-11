import numpy
import math
import random
import pygame

import Box2D

import src.utils as utils
import src.convexhull as convexhull

GAME_TITLE = "Containment"
LEVEL_DIR = "levels"

DIMS = (64, 48)
DISPLAY_SCALE_FACTOR = 4
EXTRA_SCREEN_HEIGHT = 48
SCREEN_DIMS = (DISPLAY_SCALE_FACTOR * DIMS[0],
               DISPLAY_SCALE_FACTOR * DIMS[1] + EXTRA_SCREEN_HEIGHT)

BOX2D_SCALE_FACTOR = 1

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

    player = None
    barrel = None

    heart_icon = None
    skull_icon = None
    bar_empty = None
    bar_full = None

    @staticmethod
    def load(filepath):
        def scale(surf, factor):
            return pygame.transform.scale_by(surf, (factor, factor))
        img = pygame.image.load(filepath).convert_alpha()

        y = 0
        Spritesheet.player = img.subsurface([0, y, 32, 32])
        Spritesheet.barrel = img.subsurface([32, y, 32, 32])

        y += 32
        bar_sc = 2
        Spritesheet.heart_icon = scale(img.subsurface([0, y, 17, 17]), bar_sc)
        Spritesheet.skull_icon = scale(img.subsurface([17 + 46, y, 17, 17]), bar_sc)
        Spritesheet.bar_empty = scale(img.subsurface([17, y, 46, 17]), bar_sc)
        Spritesheet.bar_full = scale(img.subsurface([15, y + 16, 50, 17]), bar_sc)


class Entity:

    def __init__(self, xy, dims=(1, 1)):
        self.xy = xy  # top left corner
        self.dims = dims
        self.body = None

    def get_rect(self):
        return (self.xy[0], self.xy[1], self.dims[0], self.dims[1])

    def get_center(self):
        return (self.xy[0] + self.dims[0] / 2, self.xy[1] + self.dims[1] / 2)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        return None

    def update(self, dt, level, **kwargs):
        pass

    def render(self, surf, color=(255, 255, 255)):
        rect = self.get_rect()
        pygame.draw.rect(surf, color, (int(rect[0] * DISPLAY_SCALE_FACTOR),
                                       int(rect[1] * DISPLAY_SCALE_FACTOR),
                                       max(1, round(rect[2] * DISPLAY_SCALE_FACTOR)),
                                       max(1, round(rect[3] * DISPLAY_SCALE_FACTOR))))

    def get_center_xy_on_screen(self):
        cx, cy = self.get_center()
        return (DISPLAY_SCALE_FACTOR * cx, DISPLAY_SCALE_FACTOR * cy)

    def convert_to_screen_pt(self, xy):
        return DISPLAY_SCALE_FACTOR * xy[0], DISPLAY_SCALE_FACTOR * xy[1]

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


def make_dynamic_polygon_body(world, polygon, color=(125, 255, 125)) -> Box2D.b2Body:
    avg_x = sum(xy[0] * BOX2D_SCALE_FACTOR for xy in polygon) / len(polygon)
    avg_y = sum(xy[1] * BOX2D_SCALE_FACTOR for xy in polygon) / len(polygon)
    shifted_pts = [(x * BOX2D_SCALE_FACTOR - avg_x, y * BOX2D_SCALE_FACTOR - avg_y) for (x, y) in polygon]

    return world.CreateDynamicBody(
        position=(avg_x, avg_y),
        userData={
            'color': color
        },
        linearDamping=10,
        angularDamping=10,
        fixtures=Box2D.b2FixtureDef(
            shape=Box2D.b2PolygonShape(vertices=shifted_pts),
            density=1,
            restitution=0.12
        )
    )

def make_dynamic_circle_body(world, xy, radius, color=(125, 125, 125)):
    return world.CreateDynamicBody(
        position=(xy[0] * BOX2D_SCALE_FACTOR, xy[1] * BOX2D_SCALE_FACTOR),
        userData={
            'color': color
        },
        linearDamping=10,
        angularDamping=10,
        fixtures=[
            Box2D.b2FixtureDef(
                shape=Box2D.b2CircleShape(pos=(0, 0), radius=radius * BOX2D_SCALE_FACTOR),
                density=1,
                restitution=0.12
            )
        ]
    )


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

    def render(self, surf, color=(255, 0, 0)):
        center_xy = self.get_center_xy_on_screen()
        sprite = Spritesheet.barrel
        blit_xy = (center_xy[0] - sprite.get_width() // 2,
                   center_xy[1] - 3 * sprite.get_height() // 4)
        surf.blit(Spritesheet.barrel, blit_xy)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        radius = math.sqrt((self.dims[0] / 2)**2 + (self.dims[1] / 2)**2)
        return make_dynamic_circle_body(world, self.get_center(), radius, (255, 0, 0))

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


class PolygonEntity(Entity):

    def __init__(self, poly_list):
        super().__init__(poly_list[0])  # TODO calc centroid for xy

        chull = convexhull.ConvexHull()
        chull.add_all(pt for pt in poly_list)
        self.poly_list = chull.get_hull_points()

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        return make_dynamic_polygon_body(world, self.poly_list, (255, 255, 255))

    def render(self, surf, color=(255, 255, 255)):
        base_pts = [self.convert_to_screen_pt(xy) for xy in self.poly_list]
        top_pts = [(x, y - 8) for (x, y) in base_pts]

        chull = convexhull.ConvexHull()
        chull.add_all(base_pts)
        chull.add_all(top_pts)
        hull_pts = [pygame.Vector2(xy) for xy in chull.get_hull_points()]

        pygame.draw.polygon(surf, (0, 0, 0), hull_pts, width=3)
        pygame.draw.polygon(surf, (47, 47, 47), hull_pts)

        min_x = float('inf')
        max_x = -float('inf')
        for (x, y) in base_pts:
            min_x = min(x, min_x)
            max_x = max(x, max_x)

        for i in range(len(base_pts)):
            base_xy = base_pts[i]
            if min_x < base_xy[0] < max_x:
                pygame.draw.line(surf, (0, 0, 0), base_xy, top_pts[i])

        pygame.draw.polygon(surf, (90, 90, 90), top_pts)
        pygame.draw.polygon(surf, (195, 195, 195), top_pts, width=1)


class Player(ParticleAbsorber):

    def __init__(self, xy, dims=(2, 2)):
        super().__init__(xy, dims=dims)
        self.last_dir = pygame.Vector2(0, 1)

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
            self.last_dir = move_dir
        else:
            self.resolve_rect_collision_with_level(level)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        radius = math.sqrt((self.dims[0] / 2)**2 + (self.dims[1] / 2)**2)
        return make_dynamic_circle_body(world, self.get_center(), radius, (255, 0, 0))

    def render(self, surf, color=(0, 0, 255)):
        center_xy_on_screen = self.get_center_xy_on_screen()

        angle = self.last_dir.as_polar()[1]
        sprite = pygame.transform.rotate(Spritesheet.player, -angle + 90)

        blit_xy = (center_xy_on_screen[0] - sprite.get_width() // 2,
                   center_xy_on_screen[1] - sprite.get_height() // 2)
        surf.blit(sprite, blit_xy)
        surf.set_at((int(center_xy_on_screen[0]), int(center_xy_on_screen[1])), (255, 0, 0))


class Level:

    def __init__(self, size, spawn_rate=SPAWN_RATE):
        self.size = size
        self.particles: ParticleArray = ParticleArray()
        self.spawn_rate = spawn_rate
        self.spatial_hash = {}

        self.player = None
        self.absorbers = []
        self.emitters = []
        self.polygons = []

        self.geometry = pygame.Surface(size).convert()
        self.energy = numpy.zeros(size, dtype=numpy.float16)

        self.world = Box2D.b2World(gravity=(0, 0))

    def update(self, dt):
        self.update_emitters(dt)
        self.update_particles(dt)
        if self.player is not None:
            self.player.update(dt, self)
        self.update_absorbers(dt)
        self.update_polygons(dt)

        self.energy *= (1 - AMBIENT_ENERGY_DECAY_RATE * dt)

        self.world.Step(dt, 6, 2)

    def update_emitters(self, dt):
        total_weight = sum((emit.weight for emit in self.emitters), start=0)
        for emitter in self.emitters:
            rate = self.spawn_rate * emitter.weight / total_weight
            emitter.update(dt, self, rate=rate)

    def update_particles(self, dt):
        self.particles.update(dt)
        self._handle_collisions(dt)

    def update_polygons(self, dt):
        for poly in self.polygons:
            poly.update(dt, self)

    def update_absorbers(self, dt):
        for absorber in self.absorbers:
            for idx in self.all_particles_indices_in_rect(absorber.get_rect()):
                absorber.absorb(dt, self, idx)

    def add_entity(self, ent):
        ent.body = ent.build_box2d_obj(self.world)
        if isinstance(ent, ParticleEmitter):
            self.emitters.append(ent)
        if isinstance(ent, ParticleAbsorber):
            self.absorbers.append(ent)
        if isinstance(ent, Player):
            if self.player is not None:
                raise ValueError("level already has a player")
            self.player = ent
        if isinstance(ent, PolygonEntity):
            self.polygons.append(ent)

    def remove_entity(self, ent):
        if isinstance(ent, ParticleEmitter):
            self.emitters.remove(ent)
        if isinstance(ent, ParticleAbsorber):
            self.absorbers.remove(ent)
        if isinstance(ent, Player):
            self.player = None  # hope it's the same player
        if isinstance(ent, PolygonEntity):
            self.polygons.remove(ent)

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

    def render_geometry(self, surf: pygame.Surface, color=(255, 255, 255)):
        surf.fill(color)
        surf.blit(self.geometry, (0, 0), special_flags=pygame.BLEND_RGB_SUB)

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
        max_energy = PARTICLE_ENERGY * 1.5
        if max_energy > 0:
            rbuf[:] = (numpy.clip(self.energy / max_energy, 0, 1) * 255).astype(numpy.int16)

    def all_entities(self):
        seen = set()
        if self.player is not None:
            yield self.player
            seen.add(id(self.player))
        for emitter in self.emitters:
            if id(emitter) not in seen:
                yield emitter
                seen.add(id(emitter))
        for absorber in self.absorbers:
            if id(absorber) not in seen:
                yield absorber
                seen.add(id(absorber))
        for polygon in self.polygons:
            if id(polygon) not in seen:
                yield polygon
                seen.add(id(polygon))

    def render_entities(self, surf: pygame.Surface):
        for ent in self.all_entities():
            ent.render(surf)

def load_level_from_file(filename) -> Level:
    img = pygame.image.load(f'{LEVEL_DIR}/{filename}').convert()
    size = img.get_size()
    res = Level(size)

    other_colors = {}

    for y in range(size[1]):
        for x in range(size[0]):
            clr = img.get_at((x, y))
            if clr == (255, 0, 0):
                res.add_entity(ParticleEmitter((x + 0.5, y + 0.5)))
                img.set_at((x, y), (255, 255, 255))
            elif clr == (0, 0, 255):
                res.add_entity(Player((x + 1, y + 1)))
                img.set_at((x, y), (255, 255, 255))
            elif clr == (255, 255, 255) or clr == (0, 0, 0):
                pass
            else:
                if clr.rgb not in other_colors:
                    other_colors[clr.rgb] = []
                other_colors[clr.rgb].append((x, y))

    for clr in other_colors:
        res.add_entity(PolygonEntity(other_colors[clr]))

    res.geometry.blit(img, (0, 0))
    return res


def render_dose_bar(surf: pygame.Surface, rect, pcnt):
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

def screen_xy_to_world_xy(screen_xy, screen_rect, camera_rect, rounded=False):
    res = (screen_xy[0] / screen_rect[2] * camera_rect[2] + camera_rect[0],
           screen_xy[1] / screen_rect[3] * camera_rect[3] + camera_rect[1])
    return (round(res[0]), round(res[1])) if rounded else res

def world_xy_to_screen_xy(world_xy, screen_rect, camera_rect, rounded=False):
    res = ((world_xy[0] - camera_rect[0]) * screen_rect[2] / camera_rect[2],
           (world_xy[1] - camera_rect[1]) * screen_rect[3] / camera_rect[3])
    return (round(res[0]), round(res[1])) if rounded else res

def draw_fixture(surf, fixture: Box2D.b2Fixture, camera_rect, color=(255, 255, 255), width=1):
    if isinstance(fixture.shape, Box2D.b2CircleShape):
        pt_b2 = fixture.body.GetWorldPoint(fixture.shape.pos)
        pt = (pt_b2[0] / BOX2D_SCALE_FACTOR, pt_b2[1] / BOX2D_SCALE_FACTOR)
        surf_pt = world_xy_to_screen_xy(pt, surf.get_rect(), camera_rect)
        r = round(fixture.shape.radius / BOX2D_SCALE_FACTOR * surf.get_width() / camera_rect[2])
        pygame.draw.circle(surf, color, surf_pt, r, width=width)
        # if DRAW_PTS:
        #     pygame.draw.circle(surf, DRAW_PTS[0], surf_pt, DRAW_PTS[1])
    else:
        xform_pts_b2 = [fixture.body.GetWorldPoint(pt) for pt in fixture.shape.vertices]
        xform_pts = [(x / BOX2D_SCALE_FACTOR, y / BOX2D_SCALE_FACTOR) for (x, y) in xform_pts_b2]
        surf_pts = [world_xy_to_screen_xy(pt, surf.get_rect(), camera_rect) for pt in xform_pts]
        pygame.draw.polygon(surf, color, surf_pts, width=width)
        # if DRAW_PTS:
        #     for surf_pt in surf_pts:
        #         pygame.draw.circle(surf, DRAW_PTS[0], surf_pt, DRAW_PTS[1])

def draw_body(surf, body, camera_rect, color=None, width=1):
    if color is None:
        if body.userData is not None and 'color' in body.userData:
            color = body.userData['color']
        else:
            color = (255, 255, 255)
    for fix in body.fixtures:
        draw_fixture(surf, fix, camera_rect, color=color, width=width)

def render_box2d_world(surf, world: Box2D.b2World, camera_rect):
    # bounds = [0, 0, DIMS[0], DIMS[1]]
    # world_bounds = [0, 0, DIMS[0] * BOX2D_SCALE_FACTOR, DIMS[1] * BOX2D_SCALE_FACTOR]
    # pygame.draw.rect()

    for body in world.bodies:
        draw_body(surf, body, camera_rect)


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

        if KEYS_HELD_THIS_FRAME[pygame.K_b]:
            level_screen_rect = (0, 0, DIMS[0] * DISPLAY_SCALE_FACTOR, DIMS[1] * DISPLAY_SCALE_FACTOR)
            render_box2d_world(screen.subsurface(level_screen_rect), level.world,
                               [0, 0, DIMS[0] * BOX2D_SCALE_FACTOR,
                                DIMS[1] * BOX2D_SCALE_FACTOR])
        else:
            if KEYS_HELD_THIS_FRAME[pygame.K_p]:
                level.render_geometry(rad_surf, color=(35, 40, 50))
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
                                       f"N={len(level.particles)}, pE={level.player.energy_accum:.2f})")

        dt = clock.tick() / 1000
        frm_cnt += 1

