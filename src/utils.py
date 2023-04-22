import typing

import pygame
import pygame._sdl2 as sdl2

import sys, os


# from (my own) https://github.com/davidpendergast/pygame-utils/blob/main/rainbowize.py
def make_fancy_scaled_display(
        size,
        scale_factor=0.,
        extra_flags=0,
        outer_fill_color=None,
        smooth: bool = None) -> pygame.Surface:
    """Creates a SCALED pygame display with some additional customization options.

        Args:
            size: The base resolution of the display surface.
            extra_flags: Extra display flags (aside from SCALED) to give the display, e.g. pygame.RESIZABLE.
            scale_factor: The initial scaling factor for the window.
                    For example, if the display's base size is 64x32 and this arg is 5, the window will be 320x160
                    in the physical display. If this arg is 0 or less, the window will use the default SCALED behavior
                    of filling as much space as the computer's display will allow.
                    Non-integer values greater than 1 can be used here too. Positive values less than 1 will act like 1.
            outer_fill_color: When the display surface can't cleanly fill the physical window with an integer scale
                    factor, a solid color is used to fill the empty space. If provided, this param sets that color
                    (otherwise it's black by default).
            smooth: Whether to use smooth interpolation while scaling.
                    If True: The environment variable PYGAME_FORCE_SCALE will be set to 'photo', which according to
                        the pygame docs, "makes the scaling use the slowest, but highest quality anisotropic scaling
                        algorithm, if it is available." This gives a smoother, blurrier look.
                    If False: PYGAME_FORCE_SCALE will be set to 'default', which uses nearest-neighbor interpolation.
                    If None: PYGAME_FORCE_SCALE is left unchanged, resulting in nearest-neighbor interpolation (unless
                        the variable has been set beforehand). This is the default behavior.
        Returns:
            The display surface.
    """

    # if smooth is not None:
    #     # must be set before display.set_mode is called.
    #     os.environ['PYGAME_FORCE_SCALE'] = 'photo' if smooth else 'default'

    # create the display in "hidden" mode, because it isn't properly sized yet
    res = pygame.display.set_mode(size, pygame.SCALED | extra_flags | pygame.HIDDEN)

    window = sdl2.Window.from_display_module()

    # due to a bug, we *cannot* let this Window object get GC'd
    # https://github.com/pygame-community/pygame-ce/issues/1889
    globals()["sdl2_Window_ref"] = window  # so store it somewhere safe...

    # resize the window to achieve the desired scale factor
    if scale_factor > 0:
        initial_scale_factor = max(scale_factor, 1)  # scale must be >= 1
        window.size = (int(size[0] * initial_scale_factor),
                       int(size[1] * initial_scale_factor))
        window.position = sdl2.WINDOWPOS_CENTERED  # recenter it too

    # set the out-of-bounds color
    if outer_fill_color is not None:
        renderer = sdl2.Renderer.from_window(window)
        renderer.draw_color = pygame.Color(outer_fill_color)

    # show the window (unless the HIDDEN flag was passed in)
    if not (pygame.HIDDEN & extra_flags):
        window.show()

    return res


def lerp(start, end, t, clamp=True):
    lower = min(start, end)
    upper = max(start, end)
    return max(lower, min(upper, start + t * (end - start)))


def int_mults(v, a):
    return tuple(int(v[i] * a) for i in range(len(v)))

def int_sub(v1, v2):
    return tuple(int(v1[i] - v2[i]) for i in range(len(v1)))


def int_lerp(i1, i2, t):
    return round(i1 + t * (i2 - i1))


def int_lerps(v1, v2, t):
    return tuple(int_lerp(v1[i], v2[i], t) for i in range(len(v1)))


def tint_color(c1, c2, strength, max_shift=255):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    return (int(r1 + min(max_shift, strength * (r2 - r1))),
            int(g1 + min(max_shift, strength * (g2 - g1))),
            int(b1 + min(max_shift, strength * (b2 - b1))))


def dist2(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


def bounding_box(pts):
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')
    for (x, y) in pts:
        min_x = min(x, min_x)
        max_x = max(x, max_x)
        min_y = min(y, min_y)
        max_y = max(y, max_y)
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def time_to_str(seconds=0., minutes=0., hours=0.,  # NOQA
                decimals: typing.Union[int, typing.Tuple[int, int]] = (1, 3),
                show_hours_as_minutes=False) -> str:
    """Formats a quantity of elapsed time into a human-readable string.

        The format will be one of the following:
            h:mm:ss[.xxx...]
               m:ss[.xxx...]
                  s[.xxx...]
        Args:
            decimals: The minimum and maximum number of decimal places to display. If a single number is given,
                      it will be interpreted as an exact number of decimal places to include.
            show_hours_as_minutes: If True, the hours places will be shown as minutes, going above 60 if needed.
    """
    total_secs = hours * 60 * 60 + minutes * 60 + seconds

    is_neg = total_secs < 0
    total_secs = abs(total_secs)

    if isinstance(decimals, tuple):
        min_dec, max_dec = decimals
    else:
        min_dec, max_dec = (decimals, decimals)

    frac = total_secs - int(total_secs)
    if max_dec <= 0:
        frac_str = ""
    else:
        frac_str = '{num:.{prec}f}'.format(num=frac, prec=max_dec)[2:]
        if len(frac_str) < min_dec:
            # lengthen the decimal part if it's too short
            frac_str = frac_str + '0' * (min_dec - len(frac_str))
        else:
            # slice off unnecessary trailing zeros
            while frac_str.endswith('0') and len(frac_str) > min_dec:
                frac_str = frac_str[:-1]
    total_secs = int(total_secs)

    s = total_secs % 60

    if not show_hours_as_minutes:
        m = (total_secs // 60) % 60
        hr = (total_secs // 3600)
    else:
        m = (total_secs // 60)
        hr = 0

    s2 = f"{s}".zfill(2)  # "2" -> "02"
    m2 = f"{m}".zfill(2)  # "2" -> "02"

    if hr > 0:
        res = f"{hr}:{m2}:{s2}"
    elif m > 0:
        res = f"{m}:{s2}"
    else:
        res = f"{s}"

    if len(frac_str) > 0:
        res = f"{res}.{frac_str}"

    if is_neg:
        res = f"-{res}"

    return res


# yoinkers from https://stackoverflow.com/a/13790741
def res_path(filepath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, filepath)
