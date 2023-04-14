import pygame

import const
import src.sprites as sprites
import src.sounds as sounds
import src.utils as utils
import math


class SceneManager:

    def __init__(self, start: 'Scene'):
        self.active_scene = None
        self._next_scene = start

    def jump_to_scene(self, next_scene):
        self._next_scene = next_scene

    def update(self, dt):
        if self._next_scene is not None:
            if self.active_scene is not None:
                self.active_scene.on_exit()
                self.active_scene.manager = None
            self.active_scene = self._next_scene
            self.active_scene.on_start()
            self._next_scene = None

        self.active_scene.update(dt)

    def render(self, surf):
        bg_color = self.active_scene.get_bg_color()
        if bg_color is not None:
            surf.fill(bg_color)
        self.active_scene.render(surf)


class Scene:

    def __init__(self):
        self.elapsed_time = 0
        self.manager = None

    def on_start(self):
        pass

    def on_exit(self):
        pass

    def update(self, dt):
        self.elapsed_time += dt

    def render(self, surf: pygame.Surface):
        pass

    def get_bg_color(self):
        return (0, 0, 0)

    def get_caption_info(self):
        return {}


class TitleScene(Scene):

    def __init__(self, title_img=None, title_y_pos=0.333, bg_img=None,
                 overlay_top_imgs=(), overlay_bottom_imgs=()):
        super().__init__()
        self.title_img = title_img
        self.title_y_pos = title_y_pos
        self.overlay_top_imgs = overlay_top_imgs
        self.overlay_bottom_imgs = overlay_bottom_imgs
        self.bg_img = bg_img

    def update(self, dt):
        super().update(dt)

    def on_start(self):
        sounds.play_song(utils.res_path(const.MAIN_SONG))

    def get_overlay_y_offset(self, for_surf, side='top'):
        start, end = (16, 0)
        duration = 1.333
        return utils.lerp(start, end, self.elapsed_time / duration)

    def get_title_tint(self):
        color = (64, 0, 0)
        start, end = (1, 0)
        duration = 3.333
        strength = utils.lerp(start, end, self.elapsed_time / duration)
        return color, strength

    def get_bg_tint(self):
        color = (0, 0, 0)
        start, end = (1, 0)
        duration = 5
        wobble = 0.2 * (1 + math.cos(math.pi * 2 * self.elapsed_time / 1.666)) / 2
        strength = utils.lerp(start, end, self.elapsed_time / duration) + wobble
        return color, strength

    def render(self, surf):
        if self.bg_img is not None:
            bg_tinted = sprites.tint(self.bg_img, *self.get_bg_tint())
            surf.blit(bg_tinted, (int(surf.get_width() / 2 - bg_tinted.get_width() / 2),
                                  int(surf.get_height() / 2 - bg_tinted.get_height() / 2)))
        if self.title_img is not None:
            title_resized = sprites.resize(self.title_img, (None, 54), method='smooth')
            title_tinted = sprites.tint(title_resized, *self.get_title_tint())
            surf.blit(title_tinted, (int(surf.get_width() / 2 - title_tinted.get_width() / 2),
                                     int(surf.get_height() * self.title_y_pos - title_tinted.get_height() / 2)))

        y_offs = self.get_overlay_y_offset(surf, side='top')
        if len(self.overlay_top_imgs) == 1:
            top_overlay_resized = sprites.resize(self.overlay_top_imgs[0], (surf.get_width(),
                                                                            self.overlay_bottom_imgs[0].get_height()))
            surf.blit(top_overlay_resized, (0, -y_offs))
        elif len(self.overlay_top_imgs) == 3:
            middle_overlay_resized = sprites.resize(self.overlay_top_imgs[1],
                (surf.get_width() - self.overlay_top_imgs[0].get_width() - self.overlay_top_imgs[2].get_width(),
                 self.overlay_top_imgs[1].get_height()))
            surf.blit(middle_overlay_resized, (self.overlay_top_imgs[0].get_width(), -y_offs))
            surf.blit(self.overlay_top_imgs[0], (0, -y_offs))
            surf.blit(self.overlay_top_imgs[2], (surf.get_width() - self.overlay_top_imgs[2].get_width(), -y_offs))

        y_offs = self.get_overlay_y_offset(surf, side='bottom')
        if len(self.overlay_bottom_imgs) == 1:
            bottom_overlay_resized = sprites.resize(self.overlay_bottom_imgs[0],
                                                    (surf.get_width(), self.overlay_bottom_imgs[0].get_height()))
            surf.blit(bottom_overlay_resized, (0, surf.get_height() - bottom_overlay_resized.get_height() + y_offs))
        elif len(self.overlay_bottom_imgs) == 3:
            middle_overlay_resized = sprites.resize(self.overlay_bottom_imgs[1],
                (surf.get_width() - self.overlay_bottom_imgs[0].get_width() - self.overlay_bottom_imgs[2].get_width(),
                 self.overlay_bottom_imgs[1].get_height()))
            surf.blit(middle_overlay_resized, (self.overlay_bottom_imgs[0].get_width(),
                                               surf.get_height() - self.overlay_bottom_imgs[0].get_height() + y_offs))
            surf.blit(self.overlay_bottom_imgs[0], (0, surf.get_height() - self.overlay_bottom_imgs[0].get_height() + y_offs))
            surf.blit(self.overlay_bottom_imgs[2], (surf.get_width() - self.overlay_bottom_imgs[2].get_width(),
                                                    surf.get_height() - self.overlay_bottom_imgs[2].get_height() + y_offs))


class OptionMenuScene(TitleScene):

    FONT_CACHE = {}  # name, size -> Font

    def __init__(self, options=(), title_img=None, title_y_pos=0.333, bg_img=None,
                 overlay_top_imgs=(), overlay_bottom_imgs=()):
        super().__init__(title_img=title_img, title_y_pos=title_y_pos, bg_img=bg_img,
                         overlay_top_imgs=overlay_top_imgs, overlay_bottom_imgs=overlay_bottom_imgs)
        self.options = options
        self.sel_opt = 0

        self._cached_option_renderings = {}  # (text, color) -> Surface

        self.base_color = (247, 196, 196)
        self.sel_color = (214, 10, 10)
        self.disabled_color = (76, 72, 72)
        self.disabled_sel_color = (159, 154, 154)

    def get_font(self, size=32):
        key = ('n', size)
        if key not in OptionMenuScene.FONT_CACHE:
            OptionMenuScene.FONT_CACHE[key] = pygame.font.Font(utils.res_path(const.MAIN_FONT), size)
        return OptionMenuScene.FONT_CACHE[key]

    def opt_enabled(self, opt_name):
        return True

    def selected_opt(self, opt_name):
        sounds.play_sound(const.MENU_MOVE)

    def activate_option(self, opt_name):
        print(f"Option was activated: {opt_name}")
        sounds.play_sound(const.MENU_SELECT)

    def update(self, dt):
        super().update(dt)
        if len(self.options) > 0:
            dy = 0
            if pygame.K_UP in const.KEYS_PRESSED_THIS_FRAME or pygame.K_w in const.KEYS_PRESSED_THIS_FRAME:
                dy -= 1
            if pygame.K_DOWN in const.KEYS_PRESSED_THIS_FRAME or pygame.K_s in const.KEYS_PRESSED_THIS_FRAME:
                dy += 1

            if dy != 0 and len(self.options) > 1:
                self.sel_opt = (self.sel_opt + dy) % len(self.options)
                self.selected_opt(self.options[self.sel_opt])

            if pygame.K_SPACE in const.KEYS_PRESSED_THIS_FRAME or pygame.K_RETURN in const.KEYS_PRESSED_THIS_FRAME:
                sel_name = self.options[self.sel_opt]
                if self.opt_enabled(sel_name):
                    self.activate_option(sel_name)
                else:
                    sounds.play_sound('error')

    def get_color(self, opt_name):
        is_sel = self.options.index(opt_name) == self.sel_opt
        is_enabled = self.opt_enabled(opt_name)
        if is_enabled:
            res = self.sel_color if is_sel else self.base_color
        else:
            res = self.disabled_sel_color if is_sel else self.disabled_color
        return utils.tint_color(res, *self.get_title_tint())

    def render_option(self, opt_name):
        color = self.get_color(opt_name)
        key = (opt_name, color)
        if key not in self._cached_option_renderings:
            self._cached_option_renderings[key] = self.get_font().render(opt_name, False, color)
        return self._cached_option_renderings[key]

    def get_options_render_offs(self, surf, size):
        x = int(surf.get_width() / 2 - size[0] / 2)
        y_max = int(surf.get_height() - size[1])
        y = min(y_max, int(surf.get_height() * 0.666 - size[1] / 2))
        return x, y

    def render(self, surf):
        super().render(surf)
        option_renders = [self.render_option(opt_name) for opt_name in self.options]
        h = sum((s.get_height() for s in option_renders), start=0)
        w = max((s.get_width() for s in option_renders), default=0)
        rect = [*self.get_options_render_offs(surf, (w, h)), w, h]

        y = 0
        for opt_img in option_renders:
            x = int(rect[0] + rect[2] / 2 - opt_img.get_width() / 2)
            surf.blit(opt_img, (x, y + rect[1]))
            y += opt_img.get_height()


class MainMenuScene(OptionMenuScene):

    def __init__(self):
        super().__init__(title_img=sprites.UiSheet.TITLES['meltdown'],
                         bg_img=sprites.UiSheet.MAIN_MENU_BG,
                         overlay_top_imgs=sprites.UiSheet.OVERLAY_TOPS['small_2x'],
                         overlay_bottom_imgs=sprites.UiSheet.OVERLAY_BOTTOMS['thin_2x'])

        self.options = ['start', 'levels', 'exit']

    def on_start(self):
        sounds.play_song(utils.res_path(const.MAIN_SONG))

