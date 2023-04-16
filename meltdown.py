import pygame

import const as const
import src.utils as utils
import src.sounds as sounds
import src.scenes as scenes
import src.gameplay as gameplay

from src.sprites import Spritesheet, UiSheet


if __name__ == "__main__":
    pygame.init()
    pygame.font.init()

    const.load_data_from_disk()
    
    screen = utils.make_fancy_scaled_display(
        const.SCREEN_DIMS,
        scale_factor=2,
        outer_fill_color=(121, 0, 0),
        extra_flags=pygame.RESIZABLE)
    pygame.display.set_caption(const.GAME_TITLE)

    sounds.initialize(const.SOUND_DIR)
    gameplay.initialize_level_list("levels/level_list.txt")

    rad_surf = pygame.Surface(const.DIMS)

    clock = pygame.time.Clock()
    dt = 0

    Spritesheet.load(utils.res_path("assets/sprites.png"))
    UiSheet.load(utils.res_path("assets"))

    scene_manager = scenes.SceneManager(scenes.MainMenuScene())

    frm_cnt = 0

    running = True
    while running and not scene_manager.should_quit:
        const.KEYS_PRESSED_THIS_FRAME.clear()
        const.KEYS_RELEASED_THIS_FRAME.clear()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                const.save_data_to_disk()
                running = False
            elif e.type == pygame.KEYDOWN:
                const.KEYS_PRESSED_THIS_FRAME.add(e.key)
                const.KEYS_HELD_THIS_FRAME.add(e.key)
            elif e.type == pygame.KEYUP:
                const.KEYS_RELEASED_THIS_FRAME.add(e.key)
                if e.key in const.KEYS_HELD_THIS_FRAME:
                    const.KEYS_HELD_THIS_FRAME.remove(e.key)

        scene_manager.update(dt)
        scene_manager.render(screen)

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

        dt = clock.tick(const.MAX_FPS) / 1000
        const.SAVE_DATA['time'] += dt  # add real time
        dt = min(dt, 1 / const.MIN_FPS)

        frm_cnt += 1

