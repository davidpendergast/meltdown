import os

import pygame
import traceback
import glob
import random

import const
import src.utils as utils

_DID_INIT = False
_SOUND_DIR = None
_SOUND_EFFECTS = {}

_PLAYING_SONG = None


def initialize(sound_dir):
    global _DID_INIT, _SOUND_DIR
    try:
        _SOUND_DIR = sound_dir
        pygame.mixer.init()
        _DID_INIT = True
    except Exception:
        traceback.print_exc()
        _DID_INIT = False


def play_sound(sound_id, volume=1):
    if _DID_INIT:
        if sound_id not in _SOUND_EFFECTS:
            _SOUND_EFFECTS[sound_id] = []
            sound_dir = utils.res_path(f"{_SOUND_DIR}")
            for filename in os.listdir(sound_dir):
                if filename.startswith(sound_id) and filename.endswith(('.wav', '.ogg')):
                    _SOUND_EFFECTS[sound_id].append(os.path.join(sound_dir, filename))

        if len(_SOUND_EFFECTS[sound_id]) > 0:
            idx = int(random.random() * len(_SOUND_EFFECTS[sound_id]))
            choice = _SOUND_EFFECTS[sound_id][idx]
            if isinstance(choice, str):
                as_sound = pygame.mixer.Sound(choice)
                _SOUND_EFFECTS[sound_id][idx] = as_sound
                choice = as_sound
            choice.set_volume(volume)
            choice.play()
        else:
            print(f"ERROR: Unrecognized sound: {sound_id}")


def play_song(filepath, volume=1):
    global _PLAYING_SONG
    if pygame.mixer.music.get_busy() and _PLAYING_SONG == filepath:
        return  # already playing
    else:
        try:
            if filepath is None:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
            else:
                    pygame.mixer.music.load(filepath)
                    pygame.mixer.music.set_volume(volume)
                    pygame.mixer.music.play(loops=-1)
        except Exception:
            traceback.print_exc()
        finally:
            _PLAYING_SONG = filepath