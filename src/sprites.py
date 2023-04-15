import pygame
import os
import const
import src.utils as utils

_MAX_CACHE_SIZE = 1000
_IMG_XFORM_CACHE = {}


def resize(img: pygame.Surface, size, method='nearest', nocache=False):
    if size[0] is None and size[1] is None:
        return img
    elif size[0] is None:
        size = (int(img.get_width() / img.get_height() * size[1]), size[1])
    elif size[1] is None:
        size = (size[0], int(img.get_height() / img.get_width() * size[0]))

    key = (id(img), size, method)
    if len(_IMG_XFORM_CACHE) > _MAX_CACHE_SIZE:
        print(f"WARN: _IMG_XFORM_CACHE overfilled! (current item={key})")
        if const.IS_DEV:
            raise ValueError(f"Cache overfilled, probably due to a leak.")
        _IMG_XFORM_CACHE.clear()

    if key not in _IMG_XFORM_CACHE:
        if method == 'smooth':
            resized_img = pygame.transform.smoothscale(img, size)
        else:
            resized_img = pygame.transform.scale(img, size)
        if nocache:
            return resized_img
        _IMG_XFORM_CACHE[key] = resized_img

    return _IMG_XFORM_CACHE[key]


def tint(surf: pygame.Surface, color, strength: float, nocache=False):
    q = 8
    strength = (int(255 * strength) // q) * q
    strength = min(255, max(0, strength))
    if strength == 0:
        return surf
    key = (id(surf), color, strength)
    if key not in _IMG_XFORM_CACHE:
        non_tint_img = surf.copy()
        non_tint_img.fill(utils.int_mults(utils.int_sub((255, 255, 255), color), (255 - strength) / 255), special_flags=pygame.BLEND_MULT)
        tint_img = surf.copy()
        tint_img.fill(utils.int_mults(color, strength / 255), special_flags=pygame.BLEND_MULT)
        tint_img.blit(non_tint_img, (0, 0), special_flags=pygame.BLEND_ADD)

        if nocache:
            return tint_img
        else:
            _IMG_XFORM_CACHE[key] = tint_img
    return _IMG_XFORM_CACHE[key]


def sc(surf, factor):
    return pygame.transform.scale_by(surf, (factor, factor))


def fl(surf, x=False, y=True):
    return pygame.transform.flip(surf, x, y)


class Spritesheet:

    player = None
    player_dead = None

    barrel = None
    laser = None

    heart_icon = None
    skull_icon = None
    bar_empty = None
    bar_full = None

    pg_img = None

    n_values = 10
    crystals = {}  # (type, value) -> img, base_img

    @staticmethod
    def load(filepath):
        img = pygame.image.load(filepath).convert_alpha()

        y = 0
        Spritesheet.player = img.subsurface([0, y, 32, 32])
        Spritesheet.barrel = img.subsurface([32, y, 32, 32])
        Spritesheet.laser = img.subsurface([96, 32, 32, 32])

        Spritesheet.player_dead = img.subsurface([128, 32, 32, 32])

        y += 32
        bar_sc = 2
        Spritesheet.heart_icon = sc(img.subsurface([0, y, 19, 20]), bar_sc)
        Spritesheet.skull_icon = sc(img.subsurface([19 + 44, y, 19, 20]), bar_sc)
        Spritesheet.bar_empty = sc(img.subsurface([19, y, 44, 20]), bar_sc)
        Spritesheet.bar_full = sc(img.subsurface([15, y + 16, 50, 20]), bar_sc)

        for i in range(10):
            y = 64 + (i % 5) * 32
            x = 0 + 128 * (i // 5)
            base_img = img.subsurface([x + 96, y + 13, 16, 19])
            for t in range(3):
                crystal_img = img.subsurface([x + 32 * t, y, 32, 32])
                Spritesheet.crystals[(t, Spritesheet.n_values - i - 1)] = crystal_img, base_img

        pg_img_path = utils.res_path(os.path.join("assets", "pg_pride_64x64_3.png"))
        Spritesheet.pg_img = sc(pygame.image.load(pg_img_path).convert_alpha(), 2)


class UiSheet:

    OVERLAY_TOPS = {}
    OVERLAY_BOTTOMS = {}

    MAIN_MENU_BG = None
    EMPTY_BG = None
    WIN_BG = None

    TITLES = {}

    @staticmethod
    def load(asset_dir):
        overlay_sheet = pygame.image.load(os.path.join(asset_dir, "ui_overlay.png")).convert_alpha()

        UiSheet.OVERLAY_TOPS = {
            'thin': [overlay_sheet.subsurface((0, 0, 16, 16))],
            'thick': [fl(overlay_sheet.subsurface((0, 192, 48, 48))),
                      fl(overlay_sheet.subsurface((48, 224, 16, 16))),
                      fl(overlay_sheet.subsurface((208, 192, 48, 48)))],
            'small': [fl(overlay_sheet.subsurface((0, 192-48, 48, 48))),
                      fl(overlay_sheet.subsurface((48, 224-48, 16, 16))),
                      fl(overlay_sheet.subsurface((208, 192-48, 48, 48)))]
        }
        UiSheet.OVERLAY_BOTTOMS = {}
        for key in list(UiSheet.OVERLAY_TOPS.keys()):
            UiSheet.OVERLAY_TOPS[key + '_2x'] = [sc(img, 2) for img in UiSheet.OVERLAY_TOPS[key]]
        for key in UiSheet.OVERLAY_TOPS:
            UiSheet.OVERLAY_BOTTOMS[key] = [pygame.transform.flip(img, False, True) for img in UiSheet.OVERLAY_TOPS[key]]

        UiSheet.MAIN_MENU_BG = pygame.image.load(os.path.join(asset_dir, "main_menu_bg.png")).convert_alpha()
        UiSheet.EMPTY_BG = pygame.image.load(os.path.join(asset_dir, "empty_bg.png")).convert_alpha()
        UiSheet.WIN_BG = pygame.image.load(os.path.join(asset_dir, "crystal_king_bg.png")).convert_alpha()

        for filename in os.listdir(os.path.join(asset_dir, "titles")):
            if filename.endswith(".png"):
                UiSheet.TITLES[filename[:-4]] = pygame.image.load(
                    os.path.join(asset_dir, "titles", filename)).convert_alpha()

