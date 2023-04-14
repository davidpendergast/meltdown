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


ENT_ID_CNT = 0

def next_ent_id():
    global ENT_ID_CNT
    ENT_ID_CNT += 1
    return ENT_ID_CNT - 1


class Entity:

    def __init__(self, xy=(0, 0), dims=(1, 1), **kwargs):
        self.uid = next_ent_id()
        self._xy = xy  # top left corner
        self.dims = dims
        self.body = None

    def get_render_layer(self):
        return 0

    def get_rect(self):
        cx, cy = self.get_center()
        return (cx - self.dims[0] / 2, cy - self.dims[1], self.dims[0], self.dims[1])

    def get_center(self):
        if self.body is not None:
            pt_b2 = self.body.GetWorldPoint((0, 0))
            return (pt_b2[0] / const.BOX2D_SCALE_FACTOR, pt_b2[1] / const.BOX2D_SCALE_FACTOR)
        else:
            return (self._xy[0] + self.dims[0] / 2, self._xy[1] + self.dims[1] / 2)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        return None

    def update(self, dt, level, **kwargs):
        pass

    def render(self, surf):
        rect = self.get_rect()
        pygame.draw.rect(surf, "white", (int(rect[0] * const.DISPLAY_SCALE_FACTOR),
                                         int(rect[1] * const.DISPLAY_SCALE_FACTOR),
                                         max(1, round(rect[2] * const.DISPLAY_SCALE_FACTOR)),
                                         max(1, round(rect[3] * const.DISPLAY_SCALE_FACTOR))))

    def get_center_xy_on_screen(self):
        cx, cy = self.get_center()
        return (const.DISPLAY_SCALE_FACTOR * cx, const.DISPLAY_SCALE_FACTOR * cy)

    def convert_to_screen_pt(self, xy):
        return const.DISPLAY_SCALE_FACTOR * xy[0], const.DISPLAY_SCALE_FACTOR * xy[1]

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
                    self._xy = cxy
                    return True
            print(f"Failed to resolve collision at: {self._xy}")
            return False
        return True

    def __hash__(self):
        return self.uid

    def __eq__(self, other):
        return self.uid == other.uid


def make_dynamic_polygon_body(world, polygon, color=(125, 255, 125),
                              linear_damping=10.0, angular_damping=10.0, density=1.0, restitution=0.12,
                              category_bits=const.WALL_CATEGORY, mask_bits=const.ALL_OBJECTS,
                              group_index=const.NORMAL_GROUP) -> Box2D.b2Body:
    avg_x = sum(xy[0] * const.BOX2D_SCALE_FACTOR for xy in polygon) / len(polygon)
    avg_y = sum(xy[1] * const.BOX2D_SCALE_FACTOR for xy in polygon) / len(polygon)
    shifted_pts = [(x * const.BOX2D_SCALE_FACTOR - avg_x, y * const.BOX2D_SCALE_FACTOR - avg_y) for (x, y) in polygon]

    return world.CreateDynamicBody(
        position=(avg_x, avg_y),
        userData={
            'color': color
        },
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        fixtures=Box2D.b2FixtureDef(
            shape=Box2D.b2PolygonShape(vertices=shifted_pts),
            density=density,
            restitution=restitution,
            categoryBits=category_bits, maskBits=mask_bits, groupIndex=group_index
        )
    )

def make_static_polygon_body(world, polygon, color=(125, 255, 125),
                             category_bits=const.WALL_CATEGORY, mask_bits=const.ALL_OBJECTS,
                             group_index=const.NORMAL_GROUP) -> Box2D.b2Body:
    avg_x = sum(xy[0] * const.BOX2D_SCALE_FACTOR for xy in polygon) / len(polygon)
    avg_y = sum(xy[1] * const.BOX2D_SCALE_FACTOR for xy in polygon) / len(polygon)
    shifted_pts = [(x * const.BOX2D_SCALE_FACTOR - avg_x, y * const.BOX2D_SCALE_FACTOR - avg_y) for (x, y) in polygon]

    return world.CreateStaticBody(
        position=(avg_x, avg_y),
        userData={
            'color': color
        },
        fixtures=Box2D.b2FixtureDef(
            shape=Box2D.b2PolygonShape(vertices=shifted_pts),
            categoryBits=category_bits, maskBits=mask_bits, groupIndex=group_index
        )
    )

def make_dynamic_circle_body(world, xy, radius, color=(125, 125, 125),
                             linear_damping=10.0, angular_damping=10.0, density=1.0, restitution=0.12,
                             category_bits=const.WALL_CATEGORY, mask_bits=const.ALL_OBJECTS,
                             group_index=const.NORMAL_GROUP):
    return world.CreateDynamicBody(
        position=(xy[0] * const.BOX2D_SCALE_FACTOR, xy[1] * const.BOX2D_SCALE_FACTOR),
        userData={
            'color': color
        },
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        fixtures=[
            Box2D.b2FixtureDef(
                shape=Box2D.b2CircleShape(pos=(0, 0), radius=radius * const.BOX2D_SCALE_FACTOR),
                density=density,
                restitution=restitution,
                categoryBits=category_bits, maskBits=mask_bits, groupIndex=group_index
            )
        ]
    )


def all_fixtures_in_rect(world: Box2D.b2World, rect, cond=None, continue_after=lambda fix: True) \
        -> typing.Generator[Box2D.b2Fixture, None, None]:
    """Finds all fixtures in the world whose bounding boxes overlap a rectangle."""
    rect = [x * const.BOX2D_SCALE_FACTOR for x in rect]
    res_list = []

    class MyQueryCallback(Box2D.b2QueryCallback):
        def __init__(self):
            Box2D.b2QueryCallback.__init__(self)

        def ReportFixture(self, fixture):
            if cond is None or cond(fixture):
                res_list.append(fixture)
                return continue_after(fixture)
            else:
                return True

    aabb = Box2D.b2AABB(lowerBound=(rect[0], rect[1]), upperBound=(rect[0] + rect[2], rect[1] + rect[3]))
    world.QueryAABB(MyQueryCallback(), aabb)

    for fix in res_list:
        yield fix


def all_fixtures_at_point(world: Box2D.b2World, pt, exact=True, cond=None, continue_after=lambda fix: True) \
        -> typing.Generator[Box2D.b2Fixture, None, None]:
    """Finds all fixtures in the world that contain a point."""
    pt = (pt[0] * const.BOX2D_SCALE_FACTOR, pt[1] * const.BOX2D_SCALE_FACTOR)
    xform = Box2D.b2Transform()
    xform.SetIdentity()
    for fix in all_fixtures_in_rect(world, (pt[0] - 0.001, pt[1] - 0.001, 0.002, 0.002),
                                    cond=cond, continue_after=continue_after):
        if not exact or fix.shape.TestPoint(xform, fix.body.GetLocalPoint(pt)):
            yield fix


class ParticleEntity(Entity):

    def __init__(self, xy, radius=0.25, velocity=const.PARTICLE_VELOCITY, energy=const.PARTICLE_ENERGY):
        super().__init__(xy=xy, dims=(0, 0))
        self.xy = xy
        self.radius = radius
        self.velocity = velocity
        self.energy = energy

        self.t = 0
        self.duration = const.PARTICLE_DURATION

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        body = make_dynamic_circle_body(
            world, self.xy, self.radius, (255, 0, 0),
            linear_damping=0, angular_damping=0, density=0.001, restitution=1,
            category_bits=const.PARTICLE_CATEGORY,
            mask_bits=const.WALL_CATEGORY|const.PLAYER_CATEGORY|const.CRYSTAL_CATEGORY,
            group_index=const.PARTICLE_GROUP
        )
        angle = random.random() * math.pi * 2
        body.linearVelocity = Box2D.b2Vec2(math.cos(angle) * self.velocity * const.BOX2D_SCALE_FACTOR,
                                           math.sin(angle) * self.velocity * const.BOX2D_SCALE_FACTOR)
        return body

    def update(self, dt, level, **kwargs):
        self.t += dt
        if self.duration < self.t or self.energy < const.PARTICLE_ENERGY / 100:
            level.remove_entity(self)

    def get_color(self):
        return tint(const.RADIATION_COLOR, (0, 0, 0), 0.666 * (1 - (self.energy / const.PARTICLE_ENERGY)))

    def render(self, surf):
        cx, cy = self.get_center_xy_on_screen()
        surf.set_at((int(cx), int(cy) - const.WALL_HEIGHT // 2), self.get_color())


class ParticleEmitter(Entity):

    def __init__(self, xy, dims=(3, 3), weight=1):
        super().__init__(xy=(xy[0] - dims[0] / 2, xy[1] - dims[1] / 2), dims=dims)
        self.weight = weight
        self.accum_t = 0
        self.cur_rate = 1

    def update(self, dt, level, **kwargs):
        super().update(dt, level)
        self.accum_t += dt
        n_to_spawn = int(self.cur_rate * self.accum_t)
        self.accum_t -= n_to_spawn / self.cur_rate

        for _ in range(n_to_spawn):
            self.spawn_particle(level)

    def render(self, surf):
        center_xy = self.get_center_xy_on_screen()
        sprite = Spritesheet.barrel
        blit_xy = (center_xy[0] - sprite.get_width() // 2,
                   center_xy[1] - 3 * sprite.get_height() // 4)
        surf.blit(Spritesheet.barrel, blit_xy)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        radius = math.sqrt((self.dims[0] / 2)**2 + (self.dims[1] / 2)**2)
        return make_dynamic_circle_body(world, self.get_center(), radius, (255, 0, 0),
                                        category_bits=const.EMITTER_CATEGORY, mask_bits=const.SOLID_OBJECTS,
                                        group_index=const.PARTICLE_GROUP)

    def spawn_particle(self, level: 'Level', n=1):
        c_xy = self.get_center()
        for _ in range(n):
            level.add_entity(ParticleEntity(c_xy))


class ParticleAbsorber(Entity):

    def __init__(self, xy=(0, 0), energy_limit=0, **kwargs):
        super().__init__(xy=xy, **kwargs)
        self.energy_accum = 0
        self.energy_limit = energy_limit
        self.energy_overfill_limit = 1.333
        self.decay_rate = const.AMBIENT_ENERGY_DECAY_RATE

    def absorb_particle(self, p_ent: ParticleEntity):
        to_absorb = p_ent.energy * const.ENERGY_TRANSFER_ON_COLLISION
        p_ent.energy -= to_absorb
        self.energy_accum += to_absorb

    def get_energy_pcnt(self):
        if self.energy_limit <= 0:
            return 0
        else:
            return max(0.0, min(1.0, self.energy_accum / self.energy_limit))

    def update(self, dt, level, **kwargs):
        super().update(dt, level, **kwargs)

        if self.body is not None:
            for contact_edge in self.body.contacts:
                contact = contact_edge.contact
                other_ent = contact.fixtureB.body.userData['entity']
                if isinstance(other_ent, ParticleEntity):
                    self.absorb_particle(other_ent)

        self.energy_accum *= (1 - const.AMBIENT_ENERGY_DECAY_RATE * dt)
        if self.energy_accum >= self.energy_overfill_limit * self.energy_limit:
            self.energy_accum = max(0.0, self.energy_overfill_limit * self.energy_limit)


class CrystalEntity(ParticleAbsorber):

    def __init__(self, xy, crystal_type=-1, energy_limit=const.CRYSTAL_LIMIT, **kwargs):
        super().__init__(xy=xy, dims=(3, 3), energy_limit=energy_limit, **kwargs)
        self.crystal_type = int(3 * random.random()) if crystal_type < 0 else crystal_type

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        x, y = self._xy
        w, h = self.dims
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return make_static_polygon_body(world, pts, color=(0, 255, 0),
                                        category_bits=const.WALL_CATEGORY, mask_bits=const.ALL_OBJECTS)

    def update(self, dt, level, **kwargs):
        was_full = self.get_energy_pcnt() >= 1
        super().update(dt, level)

        if not was_full and self.get_energy_pcnt() >= 1:
            level.add_entity(AnimationEntity(self.get_center(), duration=0.5))

    def render(self, surf):
        cx, cy = self.get_center_xy_on_screen()
        fill_level = int(self.get_energy_pcnt() * Spritesheet.n_values - 0.0001)
        spr, base_spr = Spritesheet.crystals[(self.crystal_type, fill_level)]

        surf.blit(base_spr, (cx - base_spr.get_width() // 2, cy - int(8 * base_spr.get_height() / 19)))
        surf.blit(spr, (cx - spr.get_width() // 2, cy - spr.get_height() + 4))


def tint(c1, c2, strength, max_shift=255):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return (int(r1 + min(max_shift, strength * (r2 - r1))),
            int(g1 + min(max_shift, strength * (g2 - g1))),
            int(b1 + min(max_shift, strength * (b2 - b1))))


class WallEntity(Entity):

    def __init__(self, poly_list, **kwargs):
        super().__init__(**kwargs)

        # chull = convexhull.ConvexHull(points=poly_list, check_colinear=True)
        #self._poly_list = chull.get_hull_points()
        self._poly_list = convexhull.convexHull(poly_list)

    def __repr__(self):
        return f"{type(self).__name__}({self._poly_list})"

    def get_pts(self):
        if self.body is not None:
            b2_pts = [self.body.GetWorldPoint(pt) for pt in self.body.fixtures[0].shape.vertices]
            return [(x / const.BOX2D_SCALE_FACTOR, y / const.BOX2D_SCALE_FACTOR) for (x, y) in b2_pts]
        else:
            return self._poly_list

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        return make_static_polygon_body(world, self._poly_list, (255, 255, 255))

    def tint_color(self, c):
        return tint(c, (0, 0, 0), 0.666)

    def render(self, surf):
        base_pts = [self.convert_to_screen_pt(xy) for xy in self.get_pts()]
        top_pts = [(x, y - const.WALL_HEIGHT) for (x, y) in base_pts]

        hull_pts = [pygame.Vector2(xy) for xy in convexhull.convexHull(base_pts + top_pts)]

        tnt = self.tint_color

        pygame.draw.polygon(surf, (0, 0, 0), hull_pts, width=3)
        pygame.draw.polygon(surf, tnt((47, 47, 47)), hull_pts)

        min_x = float('inf')
        max_x = -float('inf')
        for (x, y) in base_pts:
            min_x = min(x, min_x)
            max_x = max(x, max_x)

        for i in range(len(base_pts)):
            base_xy = base_pts[i]
            if min_x < base_xy[0] < max_x:
                pygame.draw.line(surf, (0, 0, 0), base_xy, top_pts[i])

        pygame.draw.polygon(surf, tnt((90, 90, 90)), top_pts)
        pygame.draw.polygon(surf, tnt((195, 195, 195)), top_pts, width=1)


class PolygonEntity(WallEntity, ParticleAbsorber):

    def __init__(self, poly_list, energy_limit=0, **kwargs):
        super().__init__(poly_list=poly_list, energy_limit=energy_limit, **kwargs)

    def tint_color(self, c):
        tint_color = (255, 0, 0)
        tint_strength = self.get_energy_pcnt()
        return tint(c, tint_color, tint_strength, max_shift=64)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        res = make_dynamic_polygon_body(world, self._poly_list, (255, 255, 255))
        self.energy_limit = float(res.mass * const.WALL_LIMIT_PER_KG)
        return res


class Player(ParticleAbsorber):

    def __init__(self, xy, dims=(2, 2)):
        super().__init__(xy=xy, dims=dims, energy_limit=const.PLAYER_LIMIT)
        self.last_dir = pygame.Vector2(0, 1)
        self.grab_reach = 0.666
        self.grab_joint = None

    def update(self, dt, level, **kwargs):
        super().update(dt, level)
        move_dir = pygame.Vector2()

        keys = const.KEYS_HELD_THIS_FRAME
        if pygame.K_a in keys or pygame.K_LEFT in keys:
            move_dir.x -= 1
        if pygame.K_d in keys or pygame.K_RIGHT in keys:
            move_dir.x += 1
        if pygame.K_w in keys or pygame.K_UP in keys:
            move_dir.y -= 1
        if pygame.K_s in keys or pygame.K_DOWN in keys:
            move_dir.y += 1
        if move_dir.magnitude() > 0:
            move_dir.scale_to_length(const.MOVE_SPEED)
            self.body.linearVelocity = (move_dir * const.BOX2D_SCALE_FACTOR).xy
            self.last_dir = move_dir
        else:
            self.resolve_rect_collision_with_level(level)

        if pygame.K_SPACE in const.KEYS_PRESSED_THIS_FRAME or \
                (pygame.K_SPACE in const.KEYS_HELD_THIS_FRAME and self.grab_joint is None):
            grab_pt = pygame.Vector2(self.get_center()) + pygame.Vector2(self.last_dir).normalize() * (self.grab_reach + self.dims[0] / 2)
            fix = [f for f in all_fixtures_at_point(level.world, grab_pt)
                   if not isinstance(f.body.userData['entity'], ParticleEntity)]
            if self.grab_joint is not None:
                level.world.DestroyJoint(self.grab_joint)
                self.grab_joint = None
            if len(fix) > 0:
                c_xy = (i * const.BOX2D_SCALE_FACTOR for i in self.get_center())
                o_xy = (i * const.BOX2D_SCALE_FACTOR for i in grab_pt)
                self.grab_joint = level.world.CreateDistanceJoint(
                    bodyA=self.body,
                    bodyB=fix[0].body,  # TODO calc best fixt to grab
                    anchorA=Box2D.b2Vec2(*c_xy),
                    anchorB=Box2D.b2Vec2(*o_xy),
                    collideConnected=True)
                level.add_entity(AnimationEntity(grab_pt, color=(225, 225, 225), radius=8, duration=0.25))
                sounds.play_sound('click')
        elif pygame.K_SPACE in const.KEYS_RELEASED_THIS_FRAME:
            if self.grab_joint is not None:
                level.world.DestroyJoint(self.grab_joint)
                self.grab_joint = None

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        radius = math.sqrt((self.dims[0] / 2)**2 + (self.dims[1] / 2)**2)
        return make_dynamic_circle_body(world, self.get_center(), radius, (255, 0, 0),
                                        category_bits=const.PLAYER_CATEGORY, mask_bits=const.SOLID_OBJECTS)

    def get_display_angle(self) -> float:
        if self.grab_joint is not None:
            anchor_a = (self.grab_joint.anchorA.x / const.BOX2D_SCALE_FACTOR,
                        self.grab_joint.anchorA.y / const.BOX2D_SCALE_FACTOR)
            anchor_b = (self.grab_joint.anchorB.x / const.BOX2D_SCALE_FACTOR,
                        self.grab_joint.anchorB.y / const.BOX2D_SCALE_FACTOR)
            vec = pygame.Vector2(anchor_b)
            vec -= anchor_a
            if vec.magnitude() > 0:
                return vec.as_polar()[1]
        return self.last_dir.as_polar()[1]

    def render(self, surf):
        center_xy_on_screen = self.get_center_xy_on_screen()
        angle = self.get_display_angle()
        sprite = pygame.transform.rotate(Spritesheet.player, -angle + 90)
        blit_xy = (center_xy_on_screen[0] - sprite.get_width() // 2,
                   center_xy_on_screen[1] - sprite.get_height() // 2)
        surf.blit(sprite, blit_xy)


class AnimationEntity(Entity):

    def __init__(self, xy, radius=8, color=(255, 255, 255), duration=1.0):
        super().__init__(xy=xy, dims=(0, 0))
        self.t = 0
        self.radius = radius
        self.color = color
        self.duration = duration

    def get_render_layer(self):
        return 5

    def get_prog(self) -> float:
        if self.duration > 0:
            return min(1.0, max(0.0, self.t / self.duration))
        else:
            return 0.0

    def update(self, dt, level, **kwargs):
        self.t += dt
        if 0 < self.duration < self.t:
            level.remove_entity(self)

    def render(self, surf):
        cx, cy = self.get_center_xy_on_screen()
        t = self.get_prog()
        pygame.draw.circle(surf, self.color, (cx, cy), self.radius * (1 - math.cos(t * math.pi * 2)), width=1)


class Level:

    def __init__(self, size, spawn_rate=const.SPAWN_RATE):
        self.size = size
        self.particles: ParticleArray = ParticleArray()
        self.spawn_rate = spawn_rate
        self.spatial_hash = {}

        self.player = None
        self.entities = set()

        self._ents_to_add = set()
        self._ents_to_remove = set()

        self.geometry = pygame.Surface(size).convert()
        self.energy = numpy.zeros(size, dtype=numpy.float16)

        self.world = Box2D.b2World(gravity=(0, 0))

    def update(self, dt):
        self.world.Step(dt, 6, 2)

        self.update_entities(dt)
        self.energy *= (1 - const.AMBIENT_ENERGY_DECAY_RATE * dt)

        for ent in self._ents_to_add:
            self.add_entity(ent, immediately=True)
        self._ents_to_add.clear()
        for ent in self._ents_to_remove:
            self.remove_entity(ent, immediately=True)
        self._ents_to_remove.clear()

    def update_entities(self, dt):
        emitters = []
        absorbers = []
        for ent in self.entities:
            if isinstance(ent, ParticleEmitter):
                emitters.append(ent)
            if isinstance(ent, ParticleAbsorber):
                absorbers.append(ent)

        self.update_emitters(dt, emitters)
        self.update_particles(dt)

        for ent in self.entities:
            ent.update(dt, self)

        self.update_absorbers(dt, absorbers)

    def update_emitters(self, dt, emitters):
        total_weight = sum((emit.weight for emit in emitters), start=0)
        for emitter in emitters:
            emitter.cur_rate = self.spawn_rate * emitter.weight / total_weight

    def update_particles(self, dt):
        self.particles.update(dt)
        self._handle_collisions(dt)

    def update_absorbers(self, dt, absorbers):
        for absorber in absorbers:
            for idx in self.all_particles_indices_in_rect(absorber.get_rect()):
                absorber.absorb(dt, self, idx)

    def add_entity(self, ent, immediately=False):
        if immediately:
            if ent not in self.entities:
                # print(f"Adding entity: {ent}")
                ent.body = ent.build_box2d_obj(self.world)
                if ent.body is not None:
                    ent.body.userData['entity'] = ent
                self.entities.add(ent)

                if isinstance(ent, Player):
                    if self.player is not None:
                        raise ValueError("level already has a player")
                    self.player = ent
        else:
            self._ents_to_add.add(ent)

    def remove_entity(self, ent, immediately=False):
        if immediately:
            if ent in self.entities:
                self.entities.remove(ent)
                if ent.body is not None:
                    self.world.DestroyBody(ent.body)
                if isinstance(ent, Player):
                    self.player = None  # hope it's the same player
        else:
            self._ents_to_remove.add(ent)

    def _handle_collisions(self, dt):
        self.spatial_hash.clear()
        for idx in self.particles.all_active_indices():
            x = int(self.particles.x[idx])
            y = int(self.particles.y[idx])
            if self.is_valid((x, y)):
                geom = self.get_geom_at((x, y))
                if geom > 0 and random.random() < dt * 1000:
                    energy_lost = const.ENERGY_TRANSFER_ON_COLLISION * self.particles.energy[idx]
                    self.energy[x][y] += energy_lost
                    self.particles.energy[idx] -= energy_lost

                    new_angle = random.random() * 2 * math.pi
                    self.particles.vx[idx] = const.PARTICLE_VELOCITY * math.cos(new_angle)
                    self.particles.vy[idx] = const.PARTICLE_VELOCITY * math.sin(new_angle)
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
        max_energy = const.PARTICLE_ENERGY * 1.5
        if max_energy > 0:
            rbuf[:] = (numpy.clip(self.energy / max_energy, 0, 1) * 255).astype(numpy.int16)

    def all_entities(self):
        for ent in self.entities:
            yield ent

    def render_entities(self, surf: pygame.Surface):
        all_ents = [ent for ent in self.all_entities()]
        all_ents.sort(key=lambda e: e.get_center()[1] + 10_000 * e.get_render_layer())
        for ent in all_ents:
            ent.render(surf)

def load_level_from_file(filename) -> Level:
    img = pygame.image.load(f'{const.LEVEL_DIR}/{filename}').convert()
    size = img.get_size()
    res = Level(size)

    other_colors = {}

    for y in range(size[1]):
        for x in range(size[0]):
            clr = img.get_at((x, y))
            if clr == (255, 0, 0):
                res.add_entity(ParticleEmitter((x + 0.5, y + 0.5)), immediately=True)
                img.set_at((x, y), (255, 255, 255))
            elif clr == (0, 0, 255):
                res.add_entity(Player((x + 1, y + 1)), immediately=True)
                img.set_at((x, y), (255, 255, 255))
            elif clr == (0, 255, 0):
                res.add_entity(CrystalEntity((x + 0.5, y + 0.5)), immediately=True)
                img.set_at((x, y), (255, 255, 255))
            elif clr == (255, 255, 255):
                pass  # air
            else:
                if clr.rgb not in other_colors:
                    other_colors[clr.rgb] = []
                other_colors[clr.rgb].append((x + 0.5, y + 0.5))
                img.set_at((x, y), (255, 255, 255))

    for clr in other_colors:
        if clr[0] == clr[1] == clr[2]:
            res.add_entity(WallEntity(other_colors[clr]), immediately=True)
        else:
            res.add_entity(PolygonEntity(other_colors[clr]), immediately=True)

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
        pt = (pt_b2[0] / const.BOX2D_SCALE_FACTOR, pt_b2[1] / const.BOX2D_SCALE_FACTOR)
        surf_pt = world_xy_to_screen_xy(pt, surf.get_rect(), camera_rect)
        r = round(fixture.shape.radius / const.BOX2D_SCALE_FACTOR * surf.get_width() / camera_rect[2])
        pygame.draw.circle(surf, color, surf_pt, r, width=width)
        # if DRAW_PTS:
        #     pygame.draw.circle(surf, DRAW_PTS[0], surf_pt, DRAW_PTS[1])
    else:
        xform_pts_b2 = [fixture.body.GetWorldPoint(pt) for pt in fixture.shape.vertices]
        xform_pts = [(x / const.BOX2D_SCALE_FACTOR, y / const.BOX2D_SCALE_FACTOR) for (x, y) in xform_pts_b2]
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
    for body in world.bodies:
        draw_body(surf, body, camera_rect)


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

    rad_surf = pygame.Surface(const.DIMS)

    clock = pygame.time.Clock()
    dt = 0

    level = load_level_from_file("test.png")
    Spritesheet.load(utils.res_path("assets/sprites.png"))
    UiSheet.load(utils.res_path("assets"))

    scene_manager = scenes.SceneManager(scenes.MainMenuScene())

    frm_cnt = 0

    running = True
    while running:
        const.KEYS_PRESSED_THIS_FRAME.clear()
        const.KEYS_RELEASED_THIS_FRAME.clear()
        for e in pygame.event.get():
            if e.type == pygame.QUIT or \
                    (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                const.KEYS_PRESSED_THIS_FRAME.add(e.key)
                const.KEYS_HELD_THIS_FRAME.add(e.key)
                if e.key == pygame.K_r:
                    level = load_level_from_file("test.png")
            elif e.type == pygame.KEYUP:
                const.KEYS_RELEASED_THIS_FRAME.add(e.key)
                if e.key in const.KEYS_HELD_THIS_FRAME:
                    const.KEYS_HELD_THIS_FRAME.remove(e.key)

        scene_manager.update(dt)
        scene_manager.render(screen)

        # level.update(dt)
        #
        # screen.fill("black")
        # rad_surf.fill("black")
        #
        # if pygame.K_b in const.KEYS_HELD_THIS_FRAME:
        #     level_screen_rect = (0, 0, const.DIMS[0] * const.DISPLAY_SCALE_FACTOR,
        #                          const.DIMS[1] * const.DISPLAY_SCALE_FACTOR)
        #     render_box2d_world(screen.subsurface(level_screen_rect), level.world,
        #                        [0, 0, const.DIMS[0] * const.BOX2D_SCALE_FACTOR,
        #                         const.DIMS[1] * const.BOX2D_SCALE_FACTOR])
        # else:
        #     if pygame.K_p in const.KEYS_HELD_THIS_FRAME:
        #         level.render_geometry(rad_surf, color=(35, 40, 50))
        #         level.render_particles(rad_surf)
        #     else:
        #         level.render_energy(rad_surf)
        #
        #     screen.blit(pygame.transform.scale_by(rad_surf, (const.DISPLAY_SCALE_FACTOR,) * 2), (0, 0))
        #     level.render_entities(screen)
        #
        #     dose_pcnt = 1 if level.player is None else level.player.get_energy_pcnt()
        #     dose_bar_size = (2 * const.SCREEN_DIMS[0] // 3, Spritesheet.heart_icon.get_height())
        #     render_dose_bar(screen, (const.SCREEN_DIMS[0] // 2 - dose_bar_size[0] // 2,
        #                              const.SCREEN_DIMS[1] - const.EXTRA_SCREEN_HEIGHT // 2 - dose_bar_size[1] // 2,
        #                              dose_bar_size[0], dose_bar_size[1]), dose_pcnt)

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

        dt = clock.tick(120) / 1000
        dt = min(dt, 1 / const.MIN_FPS)

        frm_cnt += 1

