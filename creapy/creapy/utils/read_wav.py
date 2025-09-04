from __future__ import annotations
import numpy as np
from soundfile import read


def read_wav(path: str, normalize: bool = True, start: float = 0.0, end: float | int = -1, mono=True,
             **kwargs) -> tuple[np.ndarray, int]:
    """reads a .wav file given in the path

    Args:
        path (str): the path to the wav file

    Returns:
        ndarray: the audio data of the sound file in a numpy array
        int: the sample rate of the sound file 
    """
    data, sr = read(path, **kwargs)
    if mono is True and data.ndim > 1:
        # convert to mono
        data = data.sum(axis=1) / data.shape[1]

    max_ = max(abs(data))
    if end == -1:
        data = data[int(start*sr):]
    else:
        data = data[int(start*sr):int(end*sr)]

    if normalize is True:
        data /= max_

    return data, sr
