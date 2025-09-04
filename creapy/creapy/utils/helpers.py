from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import tgt

from ..utils import get_config
from threading import Thread


def get_creak_intervals(series: np.ndarray, dt: np.ndarray, threshold: Optional[float] = None,
                        tgt_intervals=False):
    # _config = get_config()["MODEL"]["POSTPROCESSING"]["INTERVALS"]
    _config = get_config()
    if threshold is None:
        threshold = _config['USER']["creak_threshold"]

    t_hop = _config['USER']["hop_size"]
    # minimum creak length in seconds
    T_MIN = _config["MODEL"]["POSTPROCESSING"]["INTERVALS"]["min_creak_length"]
    T_GAP = _config["MODEL"]["POSTPROCESSING"]["INTERVALS"]["max_gap"]
    # T_BLOCK = get_config()["MODEL"]["PREPROCESSING"]["block_size"]
    # minimum creak length in blocks # TODO floor division?
    N_MIN = round(T_MIN / t_hop) + 1
    N_GAP = round(T_GAP / t_hop) - 1
    # print(N_MIN, N_GAP)
    i = 0
    creak_bin = np.where(series >= threshold, 1, 0).astype(int)
    res = np.zeros(creak_bin.shape).astype(int)
    n_segment = 0
    n_gap = np.arange(1, N_GAP+1)  # arange is important here
    creak_bin = np.append(creak_bin, np.zeros(N_GAP))
    idx0: int
    for idx, c in enumerate(creak_bin[:-N_GAP]):
        if c == 0 and not n_segment:
            continue
        elif c == 1 and not n_segment:
            idx0 = idx
            n_segment += 1
        elif c == 1 and n_segment:
            n_segment += 1
        elif c == 0 and n_segment:
            # TODO idxerror
            if any(creak_bin[idx + n_gap]):
                n_segment += 1
                continue
            elif n_segment >= N_MIN:
                res[idx0:idx0+n_segment] = 1
            n_segment = 0
            idx0 = 0

    # return res
    creak_intervals = []
    x1 = 0
    while x1 < len(res) - 1:
        if res[x1] == 1:
            for x2, t1 in enumerate(res[x1 + 1:]):
                if t1 == 0:
                    creak_intervals.append((dt[x1], dt[x1 + x2]))
                    x1 += x2 + 1
                    break
        x1 += 1

    if tgt_intervals is True:
        interval_text = get_config()["PRAAT"]["interval_text"]
        return [tgt.core.Interval(start_time=iv[0], end_time=iv[1], text=interval_text) for iv in creak_intervals]

    return creak_intervals
    # print(creak_bin)
    # print()
    # print(res)

    # while i + N_MIN + N_GAP< len(creak_bin):
    #     if creak_bin[i] == 0: i+=1; continue

    #     else:
    #         segment = creak_bin[i:i+N_MIN+1]

    #         i += len(segment)

    #     pass

    creak_pos = []
    # for idx in range(0, len(series) - 1):
    #     if series[idx] < threshold and series[idx + 1] >= threshold:
    #         creak_pos.append(idx + 1)
    #     if series[idx] >= threshold and series[idx + 1] < threshold:
    #         creak_pos.append(idx)
    # creak_intervals = list(
    #     map(lambda x: (x[0], x[1]), zip(*[iter(dt[creak_pos])] * 2)))

    # for i, interval in enumerate(creak_intervals):
    #     if interval[0] == interval[1] or abs(interval[0]-interval[1]) < _config["min_creak_length"]:
    #         creak_intervals.pop(i)

    # return creak_intervals


def get_time_vector(series: np.ndarray, sr: int, t0: float = 0):

    config_ = get_config()['USER']
    N = config_["block_size"]
    R = config_["hop_size"]

    # dt = np.arange(N / 2 * 1 / sr, (series.shape[0] + 1) * R / sr,
    #                R / sr)
    dt = N / 2 + np.linspace(
        t0, (series.shape[0] - 1) * R + t0, series.shape[0], endpoint=True
    )
    return dt


def intervals_to_textgrid(intervals: list[tgt.core.Interval],
                          textgrid_path: str,
                          result_path: str,
                          tier_name: str,
                          verbose: bool = False):

    if result_path is None:
        result_path = textgrid_path

    for encoding in ("utf-8", "utf-16"):
        try:
            textgrid = tgt.io.read_textgrid(textgrid_path, encoding=encoding)
        except UnicodeDecodeError as e:
            print(f"Error occured reading textfile:\n\n{e}")
        else:
            break

    num_tiers_including_tier_name = sum(map(lambda tier: tier_name in tier.name, textgrid.tiers))
    if num_tiers_including_tier_name:
        tier_name += f' {num_tiers_including_tier_name + 1}'
    
    interval_tier = tgt.core.IntervalTier(start_time=textgrid.start_time,
                                          end_time=textgrid.end_time,
                                          name=tier_name,
                                          objects=intervals)
    textgrid.add_tier(interval_tier)
    

    tgt.io.write_to_file(textgrid, result_path, encoding="utf-8")
    if verbose:
        print(f"Wrote textgrid at {Path(result_path).resolve()}")


def intervals_to_csv(intervals: list[tgt.core.Interval],
                     csv_dst: str):
    HEADER = "start,end,text\n"
    csv_dst = Path(csv_dst).with_suffix(".csv")
    with open(str(csv_dst), "w", encoding="utf-8") as dst:
        dst.write(HEADER)
        for iv in intervals:
            dst.write(f"{iv.start_time},{iv.end_time},{iv.text}\n")


def get_root() -> Path:
    return Path(__file__).parent.parent


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    