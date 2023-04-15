import os
import traceback

import src.readme_writer as readme_writer
import pygame
import json

IS_DEV = os.path.exists(".gitignore")
if IS_DEV:
    readme_writer.write_basic_readme()

GAME_TITLE = "Meltdown"

LEVEL_DIR = "levels"
SOUND_DIR = "assets/sounds"
MAIN_SONG = "assets/music/001-electronica-gaming-edits.ogg"
MAIN_FONT = "assets/fonts/m6x11.ttf"

DIMS = (64, 48)
DISPLAY_SCALE_FACTOR = 4
EXTRA_SCREEN_HEIGHT = 48
SCREEN_DIMS = (DISPLAY_SCALE_FACTOR * DIMS[0],
               DISPLAY_SCALE_FACTOR * DIMS[1] + EXTRA_SCREEN_HEIGHT)

CUR_PARTICLE_SIZE_IDX = 1
PARTICLE_SIZES = [0.5, 1, 2, 3, 4, 5]

MAX_FPS = 60 if IS_DEV else 120
MIN_FPS = 45

MENU_ANIM_SPEED = 3
MENU_MOVE = 'menu_move'
MENU_SELECT = 'menu_select'

BOX2D_SCALE_FACTOR = 1

KEYS_HELD_THIS_FRAME = set()
KEYS_PRESSED_THIS_FRAME = set()
KEYS_RELEASED_THIS_FRAME = set()

SAVE_DATA = {}
SAVE_DATA_FILEPATH = "userdata.json"

def load_data_from_disk():
    SAVE_DATA.clear()
    SAVE_DATA['beaten_levels'] = []
    SAVE_DATA['deaths'] = 0
    SAVE_DATA['time'] = 0

    try:
        print(f"INFO: loading data from: {SAVE_DATA_FILEPATH}")
        if os.path.exists(SAVE_DATA_FILEPATH):
            with open(SAVE_DATA_FILEPATH, 'r') as json_file:
                blob = json.load(json_file)
                if 'beaten_levels' in blob:
                    for level_name in blob['beaten_levels']:
                        SAVE_DATA['beaten_levels'].append(str(level_name))
                if 'deaths' in blob:
                    SAVE_DATA['deaths'] += int(blob['deaths'])
                if 'time' in blob:
                    SAVE_DATA['time'] += float(blob['time'])
        else:
            print("INFO: No save data found, fresh launch.")
    except Exception:
        print("ERROR: Failed to load save data.")
        traceback.print_exc()

def save_data_to_disk():
    try:
        print(f"INFO: saving data to: {SAVE_DATA_FILEPATH}")
        with open(SAVE_DATA_FILEPATH, "w") as outfile:
            json.dump(SAVE_DATA, outfile)
    except Exception:
        print("ERROR: Failed to save data.")
        traceback.print_exc()

def has_keys(key_set, keys, cond=True):
    if not cond:
        return False
    for key in keys:
        if key in key_set:
            return True
    return False


MOVE_UP_KEYS = (pygame.K_UP, pygame.K_w)
MOVE_LEFT_KEYS = (pygame.K_LEFT, pygame.K_a)
MOVE_RIGHT_KEYS = (pygame.K_RIGHT, pygame.K_d)
MOVE_DOWN_KEYS = (pygame.K_DOWN, pygame.K_s)
ACTION_KEYS = (pygame.K_SPACE, pygame.K_RETURN)
RESTART_KEYS = (pygame.K_r,)
ESCAPE_KEYS = (pygame.K_ESCAPE,)


SPAWN_RATE = 100
PARTICLE_DURATION = 7.5  # was 5

PARTICLE_VELOCITY = 20
PARTICLE_ENERGY = 400
ENERGY_TRANSFER_ON_COLLISION = 0.05

RADIATION_COLOR = (255, 0, 0)

AMBIENT_ENERGY_DECAY_RATE = 0.15

CRYSTAL_LIMIT = PARTICLE_ENERGY * 20
WALL_LIMIT_PER_KG = PARTICLE_ENERGY * 12
PLAYER_LIMIT = PARTICLE_ENERGY * 10

WALL_HEIGHT = 8

MOVE_RESOLUTION = 4
MOVE_SPEED = 16

PARTICLE_GROUP = -1
NORMAL_GROUP = 1

WALL_CATEGORY = 0x0001
PARTICLE_CATEGORY = 0x0002
EMITTER_CATEGORY = 0x0004
PLAYER_CATEGORY = 0x0008
CRYSTAL_CATEGORY = 0x0016

SOLID_OBJECTS = EMITTER_CATEGORY | PLAYER_CATEGORY | WALL_CATEGORY | CRYSTAL_CATEGORY
ALL_OBJECTS = SOLID_OBJECTS | PARTICLE_CATEGORY
