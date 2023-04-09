import pygame
import pygame._sdl2 as sdl2


# from https://github.com/davidpendergast/pygame-utils/blob/main/rainbowize.py
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