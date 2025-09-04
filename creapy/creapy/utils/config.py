from __future__ import annotations

import yaml
import ruamel.yaml
from os.path import isfile
from pathlib import Path
import sys
_RELATIVE_PATH_TO_CONFIG = "../config.yaml"
_RELATIVE_PATH_TO_USER_CONFIG = "../user_config.yaml"
_CONFIG_DIR = Path(__file__).parent / _RELATIVE_PATH_TO_CONFIG
_USER_CONFIG_DIR = Path(__file__).parent / _RELATIVE_PATH_TO_USER_CONFIG

USER_CONFIG_DIR = Path("~/.creapy/config.yaml").expanduser()

CONFIG_DIR = str(USER_CONFIG_DIR)

yaml_str = """\
audio_directory: null # Path to the directory containing audio files
textgrid_directory: null # Path to the directory containing the respective textgrid files
csv_directory: null # Path to a folder where the csv-files containing the classification results should be stored
audio_start: 0 # starttime in seconds where the audiofile should be analysed
audio_end: -1 # endtime in seconds until the audiofile should be analysed
audio_suffix: .wav # suffix of the audio-file(s)
filename_extension: null # string to append to the original textgrid filename. Creates a new file with the corresponding name.
textgrid_suffix: .TextGrid # suffix of the textgrid-file(s)
gender_model: all # The gender model chosen for creak-classification. Can be all, male or female
tier_name: creapy # tiername chosen creapy's annotations in praat
verbose: true # Verbosity of the tool
block_size: 0.04 # classification blocksize in seconds
hop_size: 0.01 # classification hopsize in seconds
creak_threshold: 0.75 # probability-threshold of the classifies where creak is classified. Can be a decimal value between 0 and 1
zcr_threshold: 0.08 # Threshold for the zero-crossing-rate pre-elimination feature
ste_threshold: 1.0e-05 # Threshold for the short-term-energy crossing rate pre-elimination feature
"""


def get_config() -> dict:
    """
    returns the configuration file as a dictionary

    Returns:
        dict: the configuration
    """

    with open(_CONFIG_DIR) as config_file:
        config: dict = ruamel.yaml.safe_load(config_file.read())

    with open(_USER_CONFIG_DIR) as user_config_file:
        _user_config: dict = ruamel.yaml.safe_load(user_config_file.read())
        
    config['USER'] = _user_config
    
    if not isfile(USER_CONFIG_DIR):
        return config

    with open(USER_CONFIG_DIR) as user_config_file:
        user_config: dict = ruamel.yaml.safe_load(user_config_file.read())

    for key in user_config.keys():
        if key not in config['USER']:
            raise ValueError(f'Invalid key found in user config: {key}')

    config['USER'].update(user_config)
    return config


def get_user_config() -> dict:
    return get_config()['USER']


def set_config(**kwargs) -> None:
    _default_config: dict = get_config()['USER']
    for key in kwargs.keys():
        if key not in _default_config:
            raise ValueError(
                f"key \"{key}\" can't be set in config, possible keys: {list(_default_config.keys())}")

    _default_config.update(kwargs)
    ruamel_yaml = ruamel.yaml.YAML()
    code = ruamel_yaml.load(yaml_str)
    code.update(_default_config)
    USER_CONFIG_DIR.parent.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG_DIR, "w") as user_config_file:
        ruamel_yaml.dump(code, user_config_file)


def reset_config() -> None:
    with open(_USER_CONFIG_DIR) as cfg_file:
        _default_config: dict = ruamel.yaml.safe_load(cfg_file.read())
    set_config(**_default_config)
