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


LEVELS = []  # loaded from level_list.txt

def level_file_to_name(lvl_file):
    return lvl_file[:-4].title()


def initialize_level_list(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        LEVELS.clear()
        for name in lines:
            if "#" in name:
                name = name[0:name.index("#")]
            name = name.strip()
            if name.endswith(".png"):
                LEVELS.append(name)


class GameplayScene(scenes.OverlayScene):

    def __init__(self, n):
        super().__init__(overlay_top_imgs=UiSheet.OVERLAY_TOPS["thin"],
                         overlay_bottom_imgs=UiSheet.OVERLAY_BOTTOMS["thick"])
        self.n = n  # level number
        self.level_name = level_file_to_name(LEVELS[n])

        filepath = utils.res_path(os.path.join(const.LEVEL_DIR, LEVELS[n]))
        self.level = Level.load_level_from_file(self, filepath)

        self.level_buf = pygame.Surface((self.level.size[0] * const.DISPLAY_SCALE_FACTOR,
                                         self.level.size[1] * const.DISPLAY_SCALE_FACTOR))
        self.insets = (0, 16, 0, 32)

        self.no_player_timer = 0
        self.spawn_death_menu_delay = 0.666

        self.crystals_satisfied_timer = 0
        self.level_win_delay = 1

    def update(self, dt):
        super().update(dt)

        if const.has_keys(const.KEYS_PRESSED_THIS_FRAME, const.RESTART_KEYS, cond=self.is_active()):
            self.manager.jump_to_scene(GameplayScene(self.n))

        # pausing
        if const.has_keys(const.KEYS_PRESSED_THIS_FRAME, const.ESCAPE_KEYS, cond=self.is_active()):
            self.manager.jump_to_scene(PauseScene(self))
            sounds.play_sound('menu_select')

        # debug level skipping
        if const.IS_DEV and const.has_keys(const.KEYS_HELD_THIS_FRAME, (pygame.K_LSHIFT,), cond=self.is_active()):
            if const.has_keys(const.KEYS_PRESSED_THIS_FRAME, (pygame.K_LEFT,)) and self.n > 0:
                self.manager.jump_to_scene(GameplayScene(self.n - 1))  # go back a level
            elif const.has_keys(const.KEYS_PRESSED_THIS_FRAME, (pygame.K_RIGHT,)) and self.n < len(LEVELS) - 1:
                self.manager.jump_to_scene(GameplayScene(self.n + 1))  # go forward a level

        if const.has_keys(const.KEYS_PRESSED_THIS_FRAME, (pygame.K_p,)):
            const.CUR_PARTICLE_SIZE_IDX = (const.CUR_PARTICLE_SIZE_IDX + 1) % len(const.PARTICLE_SIZES)

        self.level.update(dt)

        if self.is_active():
            # player died
            if self.level.player is None:
                self.no_player_timer += dt
                if self.no_player_timer >= self.spawn_death_menu_delay:
                    self.manager.jump_to_scene(YouDiedScene(self))
                    # deaths already counted in player class

            crystals = [ent for ent in self.level.all_entities() if isinstance(ent, CrystalEntity)]
            if len(crystals) == 0 or all(cry.is_fully_charged() for cry in crystals):
                self.crystals_satisfied_timer += dt
                if self.crystals_satisfied_timer >= self.level_win_delay:
                    const.SAVE_DATA['beaten_levels'].append(self.level_name)
                    const.save_data_to_disk()
                    self.manager.jump_to_scene(SuccessScene(self))
            else:
                self.crystals_satisfied_timer = max(0, self.crystals_satisfied_timer - dt)

    def render(self, screen: pygame.Surface, draw_world=True, draw_overlays=True, draw_dose_bar=True):
        level_area = [self.insets[0],
                      self.insets[1],
                      screen.get_width() - (self.insets[0] + self.insets[2]),
                      screen.get_height() - (self.insets[1] + self.insets[3])]
        if pygame.K_b in const.KEYS_HELD_THIS_FRAME:
            render_box2d_world(screen.subsurface(level_area), self.level.world,
                               [0, 0, const.DIMS[0] * const.BOX2D_SCALE_FACTOR,
                                const.DIMS[1] * const.BOX2D_SCALE_FACTOR])
        else:
            if draw_world:
                self.level_buf.fill(self.get_bg_color())
                self.level.render_entities(self.level_buf)

                screen.blit(self.level_buf, (level_area[0] + int(level_area[2] / 2 - self.level_buf.get_width() / 2),
                                             level_area[1] + int(level_area[3] / 2 - self.level_buf.get_height() / 2)))

            if draw_overlays:
                super().render(screen)  # draw overlays

            if draw_dose_bar:
                dose_pcnt = 1 if self.level.player is None else self.level.player.get_energy_pcnt()
                dose_bar_size = (2 * const.SCREEN_DIMS[0] // 3, Spritesheet.heart_icon.get_height())
                death_raise = round(12 * min(1.0, self.no_player_timer / self.spawn_death_menu_delay))
                render_dose_bar(screen, (const.SCREEN_DIMS[0] // 2 - dose_bar_size[0] // 2,
                                         const.SCREEN_DIMS[1] - const.EXTRA_SCREEN_HEIGHT // 2 - dose_bar_size[1] // 2 - death_raise,
                                         dose_bar_size[0], dose_bar_size[1]), dose_pcnt)


def render_dose_bar(surf: pygame.Surface, rect, pcnt):
    icon_w = Spritesheet.heart_icon.get_width()
    icon_h = Spritesheet.heart_icon.get_height()
    pcnt = min(1, max(0, pcnt))

    bar_rect = (rect[0] + icon_w, rect[1], rect[2] - icon_w * 2, rect[3])
    overflows = 4 * int(icon_w / 19), 2 * int(icon_w / 19)
    filled_bar_rect = (bar_rect[0] - overflows[0], rect[1],
                       bar_rect[2] + overflows[0] + overflows[1], rect[3])

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
    else:
        xform_pts_b2 = [fixture.body.GetWorldPoint(pt) for pt in fixture.shape.vertices]
        xform_pts = [(x / const.BOX2D_SCALE_FACTOR, y / const.BOX2D_SCALE_FACTOR) for (x, y) in xform_pts_b2]
        surf_pts = [world_xy_to_screen_xy(pt, surf.get_rect(), camera_rect) for pt in xform_pts]
        pygame.draw.polygon(surf, color, surf_pts, width=width)


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


class PauseScene(scenes.SceneWrapperOptionMenuScene):

    def __init__(self, inner_scene: GameplayScene):
        msg = f"Level {inner_scene.n + 1}: {inner_scene.level_name}"
        super().__init__(inner_scene,
                         update_inner=False,
                         options=('continue', 'retry', 'exit'),
                         info_text=msg,
                         title_img=UiSheet.TITLES["paused"])

    def do_continue(self):
        self.manager.jump_to_scene(self.inner_scene)

    def do_retry(self):
        if isinstance(self.inner_scene, GameplayScene):
            self.manager.jump_to_scene(GameplayScene(self.inner_scene.n))

    def render(self, surf, skip=False):
        self.inner_scene.render(surf, draw_world=True, draw_overlays=False, draw_dose_bar=True)
        self.render_overlay(surf)
        self.inner_scene.render(surf, draw_world=False, draw_overlays=True, draw_dose_bar=False)
        super().render(surf, skip=True)

    def activate_option(self, opt_name):
        super().activate_option(opt_name)
        if opt_name == 'continue':
            self.do_continue()
        elif opt_name == 'retry':
            self.do_retry()
        elif opt_name == 'exit':
            self.manager.jump_to_scene(scenes.MainMenuScene())

class YouDiedScene(scenes.SceneWrapperOptionMenuScene):

    def __init__(self, inner_scene:  GameplayScene):
        super().__init__(inner_scene, options=('retry', 'exit'), title_img=UiSheet.TITLES["you_died"])

    def do_retry(self):
        if isinstance(self.inner_scene, GameplayScene):
            self.manager.jump_to_scene(GameplayScene(self.inner_scene.n))

    def update(self, dt):
        super().update(dt)
        if const.has_keys(const.KEYS_PRESSED_THIS_FRAME, const.RESTART_KEYS):
            self.do_retry()

    def render(self, surf, skip=False):
        self.inner_scene.render(surf, draw_world=True, draw_overlays=False, draw_dose_bar=True)
        self.render_overlay(surf)
        self.inner_scene.render(surf, draw_world=False, draw_overlays=True, draw_dose_bar=False)
        super().render(surf, skip=True)

    def activate_option(self, opt_name):
        super().activate_option(opt_name)
        if opt_name == 'retry':
            self.do_retry()
        elif opt_name == 'exit':
            self.manager.jump_to_scene(scenes.MainMenuScene())


class SuccessScene(scenes.SceneWrapperOptionMenuScene):
    def __init__(self, inner_scene:  GameplayScene):
        super().__init__(inner_scene, options=('next',), title_img=UiSheet.TITLES["success"])

    def do_next(self):
        if isinstance(self.inner_scene, GameplayScene):
            if self.inner_scene.n < len(LEVELS) - 1:
                self.manager.jump_to_scene(GameplayScene(self.inner_scene.n + 1))
            else:
                self.manager.jump_to_scene(scenes.YouWinMenu())

    def activate_option(self, opt_name):
        if opt_name == 'next':
            self.do_next()

    def render(self, surf, skip=False):
        self.inner_scene.render(surf, draw_world=True, draw_overlays=False, draw_dose_bar=False)
        self.render_overlay(surf)
        self.inner_scene.render(surf, draw_world=False, draw_overlays=True, draw_dose_bar=False)
        super().render(surf, skip=True)


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

    def __init__(self, xy, radius=0.25, velocity=const.PARTICLE_VELOCITY, direction=None, energy=const.PARTICLE_ENERGY):
        super().__init__(xy=xy, dims=(0, 0))
        self.xy = xy
        self.radius = radius
        self.velocity = velocity
        if direction is None:
            angle = random.random() * math.pi * 2
            self.initial_direction = math.cos(angle), math.sin(angle)
        else:
            self.initial_direction = direction
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
        body.linearVelocity = Box2D.b2Vec2(self.initial_direction[0] * self.velocity * const.BOX2D_SCALE_FACTOR,
                                           self.initial_direction[1] * self.velocity * const.BOX2D_SCALE_FACTOR)
        return body

    def update(self, dt, level, **kwargs):
        self.t += dt

        # when the particle's getting close to the end of its duration,
        # start decaying its energy so it reaches zero when it's deleted.
        if 0.9 < self.t / self.duration < 1:
            t_remaining = self.duration - self.t
            eol_decay = min(self.energy, dt / t_remaining * self.energy)
            self.energy -= eol_decay

        if self.duration < self.t or self.energy < const.PARTICLE_ENERGY / 100:
            level.remove_entity(self)


    def get_color(self):
        max_tint = 0.666
        lightness = max(0, min(1, self.energy / const.PARTICLE_ENERGY))

        # life_prog = self.t / self.duration
        # if life_prog > 0.85:
        #     lightness *= max(0, min(1, (1 - life_prog) / (1 - 0.85)))

        return tint(const.RADIATION_COLOR, (0, 0, 0), max_tint * (1 - lightness))

    def render(self, surf):
        cx, cy = self.get_center_xy_on_screen()
        r = const.PARTICLE_SIZES[const.CUR_PARTICLE_SIZE_IDX]
        pos = (round(cx), round(cy - const.WALL_HEIGHT / 2))
        if r >= 1:
            pygame.draw.circle(surf, self.get_color(), pos, r)
        else:
            surf.set_at(pos, self.get_color())


class ParticleEmitter(Entity):

    def __init__(self, xy, dims=(3, 3), weight=1):
        super().__init__(xy=(xy[0] - dims[0] / 2, xy[1] - dims[1] / 2), dims=dims)
        self.weight = weight
        self.energy_mult = 1
        self.accum_t = 0
        self.cur_rate = 1

    def update(self, dt, level, **kwargs):
        super().update(dt, level)
        self.accum_t += dt
        n_to_spawn = int(self.cur_rate * self.accum_t)
        self.accum_t -= n_to_spawn / self.cur_rate

        energy = const.PARTICLE_ENERGY * self.energy_mult

        self.spawn_particle(level, n=n_to_spawn, energy=energy)

    def get_sprite(self):
        return Spritesheet.barrel

    def render(self, surf):
        center_xy = self.get_center_xy_on_screen()
        sprite = self.get_sprite()
        blit_xy = (center_xy[0] - sprite.get_width() // 2,
                   center_xy[1] - 3 * sprite.get_height() // 4)
        surf.blit(sprite, blit_xy)

    def build_box2d_obj(self, world) -> Box2D.b2Body:
        radius = math.sqrt((self.dims[0] / 2)**2 + (self.dims[1] / 2)**2)
        return make_dynamic_circle_body(world, self.get_center(), radius, (255, 0, 0),
                                        category_bits=const.EMITTER_CATEGORY, mask_bits=const.SOLID_OBJECTS,
                                        group_index=const.PARTICLE_GROUP)

    def spawn_particle(self, level: 'Level', n=1, energy=const.PARTICLE_ENERGY):
        c_xy = self.get_center()
        for _ in range(n):
            level.add_entity(ParticleEntity(c_xy, energy=energy))


class LaserEntity(ParticleEmitter):

    def __init__(self, xy, direction, perturb=0.01, dims=(3, 3), weight=1):
        super().__init__(xy, dims=dims, weight=weight)
        self.angle = pygame.Vector2(direction).as_polar()[1] / 360 * math.pi * 2
        self.perturb = perturb

    def get_sprite(self):
        return Spritesheet.laser

    def spawn_particle(self, level: 'Level', n=1, energy=const.PARTICLE_ENERGY):
        c_xy = self.get_center()
        for _ in range(n):
            new_angle = self.angle + 2 * (random.random() - 0.5) * self.perturb
            new_dir = (math.cos(new_angle), math.sin(new_angle))
            level.add_entity(ParticleEntity(c_xy, direction=new_dir, energy=energy))


class ParticleAbsorber(Entity):

    def __init__(self, xy=(0, 0), energy_limit=0, **kwargs):
        super().__init__(xy=xy, **kwargs)
        self.energy_accum = 0
        self.energy_limit = energy_limit
        self.energy_overfill_limit = 1.333
        self.absorb_rate = const.ENERGY_TRANSFER_ON_COLLISION
        self.decay_rate = const.AMBIENT_ENERGY_DECAY_RATE

    def absorb_particle(self, p_ent: ParticleEntity) -> float:
        to_absorb = p_ent.energy * self.absorb_rate
        p_ent.energy -= to_absorb
        self.energy_accum += to_absorb
        return to_absorb

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
        self.absorb_rate *= 1.0

    def is_fully_charged(self):
        return self.get_energy_pcnt() >= 1.0

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
            sounds.play_sound('random')
        elif was_full and self.get_energy_pcnt() < 1:
            sounds.play_sound("synth")

    def render(self, surf):
        cx, cy = self.get_center_xy_on_screen()
        fill_level = int(self.get_energy_pcnt() * (Spritesheet.n_values - 1))
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
        scene_is_active = level.scene.is_active()

        if const.has_keys(const.KEYS_HELD_THIS_FRAME, const.MOVE_LEFT_KEYS, cond=scene_is_active):
            move_dir.x -= 1
        if const.has_keys(const.KEYS_HELD_THIS_FRAME, const.MOVE_RIGHT_KEYS, cond=scene_is_active):
            move_dir.x += 1
        if const.has_keys(const.KEYS_HELD_THIS_FRAME, const.MOVE_UP_KEYS, cond=scene_is_active):
            move_dir.y -= 1
        if const.has_keys(const.KEYS_HELD_THIS_FRAME, const.MOVE_DOWN_KEYS, cond=scene_is_active):
            move_dir.y += 1
        if move_dir.magnitude() > 0:
            move_dir.scale_to_length(const.MOVE_SPEED)
            self.body.linearVelocity = (move_dir * const.BOX2D_SCALE_FACTOR).xy
            self.last_dir = move_dir

        if const.has_keys(const.KEYS_PRESSED_THIS_FRAME, const.ACTION_KEYS, cond=scene_is_active) or \
                (const.has_keys(const.KEYS_HELD_THIS_FRAME, const.ACTION_KEYS, cond=scene_is_active) and self.grab_joint is None):

            center_xy = pygame.Vector2(self.get_center())
            grab_vec = pygame.Vector2(self.last_dir).normalize() * (self.grab_reach + self.dims[0] / 2)
            grab_pts = [
                center_xy + grab_vec,
                center_xy + grab_vec.rotate(45),
                center_xy + grab_vec.rotate(-45),
            ]

            if self.grab_joint is not None:
                # this is possible if you press the grab key while
                # you have an active ghost key-hold happening.
                level.world.DestroyJoint(self.grab_joint)
                self.grab_joint = None

            for grab_pt in grab_pts:
                # important we don't grab particles or anything that might get deleted
                fix = [f for f in all_fixtures_at_point(level.world, grab_pt)
                       if not isinstance(f.body.userData['entity'], ParticleEntity)]
                if len(fix) > 0:
                    c_xy = (i * const.BOX2D_SCALE_FACTOR for i in self.get_center())
                    o_xy = (i * const.BOX2D_SCALE_FACTOR for i in grab_pt)
                    self.grab_joint = level.world.CreateDistanceJoint(
                        bodyA=self.body,
                        bodyB=fix[0].body,
                        anchorA=Box2D.b2Vec2(*c_xy),
                        anchorB=Box2D.b2Vec2(*o_xy),
                        collideConnected=True)
                    level.add_entity(AnimationEntity(grab_pt, color=(225, 225, 225), radius=8, duration=0.25))
                    sounds.play_sound('menu_select')
                    break
        elif const.has_keys(const.KEYS_RELEASED_THIS_FRAME, const.ACTION_KEYS, cond=scene_is_active):
            if self.grab_joint is not None:
                level.world.DestroyJoint(self.grab_joint)
                self.grab_joint = None

        if self.get_energy_pcnt() >= 1.0:
            level.add_entity(DeadPlayerEntity(self.get_center(), self.get_display_angle()))
            self.grab_joint = None  # this ref can segfault after the removal
            level.remove_entity(self)
            sounds.play_sound('explosion')
            const.SAVE_DATA['deaths'] += 1

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

    def absorb_particle(self, p_ent: ParticleEntity) -> float:
        pre_pcnt = self.get_energy_pcnt()
        super().absorb_particle(p_ent)
        post_pct = self.get_energy_pcnt()

        q = 5
        if int(pre_pcnt * q) != int(post_pct * q):
            sounds.play_sound('hitHurt', volume=post_pct)

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


class DeadPlayerEntity(AnimationEntity):

    def __init__(self, xy, angle):
        super().__init__(xy, duration=-1)
        self.angle = angle

    def get_render_layer(self):
        return -1  # under everything

    def render(self, surf):
        center_xy_on_screen = self.get_center_xy_on_screen()
        sprite = pygame.transform.rotate(Spritesheet.player_dead, -self.angle + 90)
        blit_xy = (center_xy_on_screen[0] - sprite.get_width() // 2,
                   center_xy_on_screen[1] - sprite.get_height() // 2)
        surf.blit(sprite, blit_xy)


class Level:

    def __init__(self, scene, size, spawn_rate=const.SPAWN_RATE):
        self.size = size
        self.scene = scene

        self.spawn_rate = spawn_rate
        self.spatial_hash = {}

        self.player = None
        self.entities = set()

        self._ents_to_add = set()
        self._ents_to_remove = set()

        self.world = Box2D.b2World(gravity=(0, 0))

    def update(self, dt):
        self.world.Step(dt, 6, 2)

        self.update_entities(dt)

        for ent in self._ents_to_add:
            self.add_entity(ent, immediately=True)
        self._ents_to_add.clear()
        for ent in self._ents_to_remove:
            self.remove_entity(ent, immediately=True)
        self._ents_to_remove.clear()

    def update_entities(self, dt):
        self.update_emit_rates()

        for ent in self.entities:
            ent.update(dt, self)

    def update_emit_rates(self):
        emitters = [ent for ent in self.entities if isinstance(ent, ParticleEmitter)]
        total_weight = sum((emit.weight for emit in emitters), start=0)
        for emitter in emitters:
            emitter.cur_rate = self.spawn_rate * emitter.weight / total_weight
            emitter.energy_mult = len(emitters)

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

    def is_valid(self, xy):
        return 0 <= xy[0] < self.size[0] and 0 <= xy[1] < self.size[1]

    def all_entities(self):
        for ent in self.entities:
            yield ent

    def render_entities(self, surf: pygame.Surface):
        all_ents = [ent for ent in self.all_entities()]
        all_ents.sort(key=lambda e: e.get_center()[1] + 10_000 * e.get_render_layer())
        for ent in all_ents:
            ent.render(surf)

    @staticmethod
    def load_level_from_file(scene, filepath) -> 'Level':
        img = pygame.image.load(filepath).convert()
        size = img.get_size()
        res = Level(scene, size)

        other_colors = {}

        laser_bases = []
        laser_tgts = []

        def is_filled_rect(pts):
            rect = utils.bounding_box(pts)
            for x in range(int(rect[0]), int(rect[0] + rect[2])):
                for y in range(int(rect[1]), int(rect[1] + rect[3])):
                    if (x, y) not in pts:
                        return False, rect
            return True, rect

        for y in range(size[1]):
            for x in range(size[0]):
                clr = img.get_at((x, y))
                if clr == (255, 0, 0):
                    res.add_entity(ParticleEmitter((x + 0.5, y + 0.5)), immediately=True)
                elif clr == (196, 0, 0):
                    laser_bases.append((x, y))
                elif clr == (196, 64, 64):
                    laser_tgts.append((x, y))
                elif clr == (0, 0, 255):
                    res.add_entity(Player((x + 1, y + 1)), immediately=True)
                elif clr == (0, 255, 0):
                    res.add_entity(CrystalEntity((x + 0.5, y + 0.5)), immediately=True)
                elif clr == (255, 255, 255):
                    pass  # air
                else:
                    if clr.rgb not in other_colors:
                        other_colors[clr.rgb] = []
                    other_colors[clr.rgb].append((x, y))

        if len(laser_tgts) > 0:
            for (x, y) in laser_bases:
                laser_tgts.sort(key=lambda pt: utils.dist2(pt, (x, y)))
                direction = (laser_tgts[0][0] - x, laser_tgts[0][1] - y)
                res.add_entity(LaserEntity((x + 0.5, y + 0.5), direction))

        for clr in other_colors:
            filled, rect = is_filled_rect(other_colors[clr])
            if filled:
                pts = [(rect[0], rect[1]),
                       (rect[0] + rect[2] + 1, rect[1]),
                       (rect[0] + rect[2] + 1, rect[1] + rect[3] + 1),
                       (rect[0], rect[1] + rect[3] + 1)]
            else:
                pts = [(x + 0.5, y + 0.5) for (x, y) in other_colors[clr]]

            if clr[0] == clr[1] == clr[2]:
                res.add_entity(WallEntity(pts), immediately=True)
            else:
                res.add_entity(PolygonEntity(pts), immediately=True)

        return res
