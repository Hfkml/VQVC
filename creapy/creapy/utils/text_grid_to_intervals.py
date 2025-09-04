from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import tgt

from .config import get_config

import warnings

def generate_sample_wavs(textgrid: tgt.core.TextGrid, audio_paths: list[str], sample_dir=None):
    config = get_config()
    if sample_dir is None:
        root = f"{Path(__file__).parent}/../.."
        sample_dir = root / Path(config['DEFAULT']['sample_dir'])
    else:
        sample_dir = Path(sample_dir)
    if not os.path.isdir(sample_dir):
        os.mkdir(sample_dir)
    tier_names = textgrid.get_tier_names()
    creak_pattern = config["PRAAT"]["creak_fstr"]
    wav_pattern = re.compile(config["DEFAULT"]["audio_re"])
    speaker_group = config["DEFAULT"]["re_speaker_group"]
    for path in audio_paths:
        data, sr = sf.read(path)
        _file = Path(path).name
        match = wav_pattern.match(_file)
        if match is None:
            warnings.warn(f"Could not find speaker in path\n\t{path}")
            continue
        # TODO variable name "speaker" is bound to speaker in config
        speaker = match.group(speaker_group)
        tier = textgrid.get_tier_by_name(creak_pattern.format(speaker))
        intervals = tier.intervals
        counter = 0
        for inter in intervals:
            times = inter.start_time, inter.end_time
            label = inter.text
            start, end = int(np.round(float(times[0]) * sr)), \
                int(np.round(float(times[1]) * sr))
            filename = f"{sample_dir}/{speaker}_{counter}_{label}.wav"
            sf.write(filename,
                     data[start:end+1],
                     sr)
            counter += 1


def read_textgrid(textgrid_path):
    try:
        textgrid = tgt.io.read_textgrid(textgrid_path, encoding="utf-8")
    except UnicodeDecodeError:
        textgrid = tgt.io.read_textgrid(textgrid_path, encoding="utf-16")
    finally:
        return textgrid
