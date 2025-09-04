from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
# import opensmile
import pandas as pd
import parselmouth as pm
import soundfile as sf
from scipy import interpolate
from scipy.signal.windows import hamming, hann, kaiser
from rich.progress import track
from ..utils import get_config


def _cpp(data: np.ndarray, sound: pm.Sound, sr: int,
         config: Optional[dict] = None) -> float:
    """Calculates the cepstral peak prominence using praat

    Args:
        data (np.ndarray): The sound data
        sr (int): The sampling rate
        config (dict, optional): The default configuration file. Defaults to None.

    Returns:
        float: The cepstral peak prominence
    """
    if config is None:
        config = get_config()

    spectrum = sound.to_spectrum()
    power_cepstrum = pm.praat.call(spectrum, 'To PowerCepstrum')

    *args, = map(config["FEATURE_EXTRACTION"]["VALUES"]["CPP"].get, [
        "fmin", "fmax", "interpolation", "qmin", "qmax", "trend_type", "fit_method"
    ])

    cpp = pm.praat.call(power_cepstrum, "Get peak prominence...", *args)
    return cpp


def _h1_h2(data: np.ndarray, sound: pm.Sound, sr: int,
           config: Optional[dict] = None) -> float:
    # if config is None:
    #     config = get_config()

    try:
        pitch = sound.to_pitch(sound.duration)
        spectrum = sound.to_spectrum()
        h1 = pitch.selected_array[0][0]
        h2 = h1 * 2

        x = np.arange(0, spectrum.nf) * spectrum.df
        y = np.sqrt(spectrum.values[0]**2 + spectrum.values[1]**2)
        f = interpolate.interp1d(x, y, 'quadratic')
        h1_amp = f(h1)
        h2_amp = f(h2)
    except:
        [h1_amp, h2_amp] = [np.nan, np.nan]

    return h1_amp - h2_amp


def _hnr(data: np.ndarray, sound: pm.Sound, sr: int,
         config: Optional[dict] = None) -> float:
    try:
        harmonicity = sound.to_harmonicity()
    except pm.PraatError:
        hnr = np.nan
    else:
        # taken from
        # https://parselmouth.readthedocs.io/en/stable/examples/batch_processing.html?highlight=harmonicity#Batch-processing-of-files
        hnr = harmonicity.values[harmonicity.values != -200].mean()
    return hnr


def _jitter(data: np.ndarray, sound: pm.Sound, sr: int,
            config: Optional[dict] = None) -> float:
    try:
        pointProcess = pm.praat.call(
            sound, "To PointProcess (periodic, cc)", 75, 500)
        local_jitter = pm.praat.call(
            pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    except:
        local_jitter = np.nan

    return local_jitter


def _shimmer(data: np.ndarray, sound: pm.Sound, sr: int,
             config: Optional[dict] = None) -> float:
    try:
        pointProcess = pm.praat.call(
            sound, "To PointProcess (periodic, cc)", 75, 500)
        local_shimmer = pm.praat.call(
            [sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except:
        local_shimmer = np.nan

    return local_shimmer


def _f0mean(data: np.ndarray, sound: pm.Sound, sr: int,
            config: Optional[dict] = None) -> float:
    try:
        pitch = sound.to_pitch(sound.duration).selected_array[0][0]
    except:
        pitch = np.nan

    return pitch


def _zcr(data: np.ndarray, sound: pm.Sound, sr: int,
         config: Optional[dict] = None):
    """
    calculates the Zero-Crossing-Rate (ZCR)
    @param data: buffer signal data with blocklength N in form of an array of shape N data num_blocks
    @param w: Window with length N
    @return Zero-Crossing-rate 

    """
    # if config is None:
    #     config = get_config()
    # w = WINDOW_MAPPING[config["FEATURE_EXTRACTION"]
    #                    ["VALUES"]["ZCR"]["window"]](data.shape[0])

    # sgn = lambda data: 1 if data >= 0 else -1
    sign_arr = np.sign(data)
    sign_arr[sign_arr == 0] = 1
    N = data.shape[0]
    return 0.5 * np.sum(np.abs(np.diff(sign_arr))) / N


def _ste(data: np.ndarray, sound: pm.Sound, sr: int,
         config: Optional[dict] = None):
    """
    calculates the short-term-energy (STE)
    @param x: buffer signal x with blocklength N in form of an array of shape N x num_blocks
    @param w: Window with length N, shape (N,)
    @return short-term-energy
    """
    # assert len(w) == x.shape[0], "Dimension Mismatch: Windowlength != blocklength"
    # assert len(x.shape) == 2, "Signal must be already buffered (2D)"
    # assert len(w.shape) == 1, "Window must be 1D"

    # if config is None:
    #     config = get_config()
    # if w is None:
    #     w = WINDOW_MAPPING[config["FEATURE_EXTRACTION"]
    #                        ["VALUES"]["STE"]["window"]](data.shape[0])
    N = data.shape[0]
    return np.sum(data ** 2) / N
    # return np.sum(np.abs(data)) / N


def get_feature_list(config: Optional[dict] = None) -> list[str]:
    if config is None:
        config = get_config()
    return [key for key, value in config["FEATURE_EXTRACTION"].items() if value is True]


def calculate_features(data: np.ndarray, sr: int,
                       return_header: bool = False,
                       config: Optional[dict] = None,
                       features: Optional[list[str]] = None) -> tuple[np.ndarray, list] | np.ndarray:
    if config is None:
        config = get_config()
    if features is None:
        features = get_feature_list(config)
    sound = pm.Sound(values=data, sampling_frequency=sr)
    result = np.array([FEATURE_MAPPING[feature](data, sound, sr, config=config)
                      for feature in features])

    return (result, features) if return_header is True else result


def blockwise_feature_calculation(data: np.ndarray, sr,
                                  feature, config: Optional[dict] = None):
    if config is None:
        config = get_config()

    sounds = [pm.Sound(values=block, sampling_frequency=sr) for block in data]
    function = FEATURE_MAPPING[feature]
    res = []
    # for block, sound in zip(track(data, description=f"Calculating {feature}"), sounds):
    #     res.append(function(block, sound, sr))
    res = [function(block, sound, sr) for block, sound in zip(data, sounds)]
    return np.array(res)


def calculate_features_for_folder(path: str,
                   file_suffix: str = ".wav",
                   features=None) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    files = list(Path(path).glob(f"**/*{file_suffix}"))
    config = get_config()
    if features is None:
        features = get_feature_list(config)
    feature_matrix = np.array(
        [calculate_features(*sf.read(file_), config=config,
                            features=features) for file_ in files]
    )
    # TODO put regex in config file
    target_vector = np.array(
        [str(x).split('.')[0].split('_')[-1] for x in files])
    return pd.DataFrame(feature_matrix, columns=features), pd.Series(target_vector)


FEATURE_MAPPING = {
    "cpp": _cpp,
    "hnr": _hnr,
    "h1h2": _h1_h2,
    "jitter": _jitter,
    "shimmer": _shimmer,
    "f0mean": _f0mean,
    "zcr": _zcr,
    "ste": _ste,
}
WINDOW_MAPPING = {
    "hann": hann,
    "kaiser": kaiser,
    "hamming": hamming,
    "rect": lambda N: np.ones((N)) / N
}
