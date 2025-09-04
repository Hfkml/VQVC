from __future__ import annotations

import numpy as np
from pandas import Series
import tgt
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

if __name__ == "__main__":  # Todo
    from creapy import get_config
else:
    try:
        from ..utils import get_config
    except ImportError as ie:
        print(ie)


# %%

def evaluation_metrics(y_true, y_pred):
    assert all((isinstance(y_pred, (np.ndarray, Series)),
               isinstance(y_true, (np.ndarray, Series))))
    assert y_pred.shape == y_true.shape
    _accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)
    print(
        f"""accuracy: {_accuracy}
        recall: {recall}
        f1: {f1},
        precision: {precision}""")
    return recall, f1, precision, confusion_matrix(y_true, y_pred)


class CreakInterval(tgt.core.Interval):

    def __init__(self, start_time, end_time, delta_start, delta_end, text=''):
        super().__init__(start_time, end_time, text)
        self.delta_start = delta_start
        self.delta_end = delta_end

    def __repr__(self):
        return u'Interval({0}, {1}, "{2}")'.format(self.start_time, self.end_time, self.text)


def evaluate(textgrid, tier_name_true, tier_name_creapy, boundary_tier_name=None,
             tier_name_evaluation=None):
    """Only possible as a function inside creapy module!"""

    TEXTGRID_PATH = textgrid
    OWN_TIER_NAME = tier_name_true
    OUTFILE = None
    CREAPY_TIER_NAME = tier_name_creapy
    # INTERVAL_LABEL = "c"
    EVALUATION_TIER_NAME = tier_name_evaluation if tier_name_evaluation is not None else "creapy-evaluation"
    for encoding in ("utf-8", "utf-16"):
        try:
            tg = tgt.io.read_textgrid(TEXTGRID_PATH, encoding=encoding)
        except UnicodeDecodeError as e:
            print(f"Error occured reading textfile:\n\n{e}")
        else:
            break

    own_tier: tgt.core.Tier = tg.get_tier_by_name(OWN_TIER_NAME)
    creapy_tier: tgt.core.Tier = tg.get_tier_by_name(CREAPY_TIER_NAME)
    # Grausliche loesung im moment
    t_min = max(own_tier.intervals[0].start_time,
                creapy_tier.intervals[0].start_time)
    t_max = min(own_tier.intervals[-1].end_time,
                creapy_tier.intervals[-1].end_time)

    own_tier = tgt.core.IntervalTier(objects=own_tier.get_annotations_between_timepoints(
        start=t_min, end=t_max, left_overlap=True, right_overlap=True))
    creapy_tier = tgt.core.IntervalTier(objects=creapy_tier.get_annotations_between_timepoints(
        start=t_min, end=t_max, left_overlap=True, right_overlap=True))
    if boundary_tier_name is not None:
        boundary_tier: tgt.core.IntervalTier = tg.get_tier_by_name(
            boundary_tier_name)
        # creapy_tier = tgt.core.IntervalTier(objects=[
        #     interval for interval in creapy_tier.intervals if not any(map(lambda x: x.text == "###" or x.text == "", boundary_tier.get_annotations_between_timepoints(
        #         start=interval.start_time, end=interval.end_time, left_overlap=True, right_overlap=True
        #     )))
        # ])

        in_boundary = []
        for interval in creapy_tier.intervals:
            ols = boundary_tier.get_annotations_between_timepoints(
                start=interval.start_time, end=interval.end_time, left_overlap=True,
                right_overlap=True
            )
            if (not all(map(lambda x: x.text in ("###", "SIL"), ols))) and bool(ols):
                in_boundary.append(interval)
        creapy_tier = tgt.core.IntervalTier(objects=in_boundary)
        in_boundary = []
        for interval in own_tier.intervals:
            ols = boundary_tier.get_annotations_between_timepoints(
                start=interval.start_time, end=interval.end_time, left_overlap=True,
                right_overlap=True
            )
            if (not all(map(lambda x: x.text in ("###", "SIL"), ols))) and bool(ols):
                in_boundary.append(interval)
        own_tier = tgt.core.IntervalTier(objects=in_boundary)

        # print([interval for interval in creapy_tier.intervals if not any(map(lambda x: x.text == "###" or x.text == "", boundary_tier.get_annotations_between_timepoints(
        #         start=interval.start_time, end=interval.end_time, left_overlap=True, right_overlap=True])

        # return np.nan, np.nan, np.nan
    overlaps = tgt.util.get_overlapping_intervals(
        own_tier,
        creapy_tier,
    )
    own_tier_copy = own_tier.get_copy_with_same_intervals_merged()
    creapy_tier_copy = creapy_tier.get_copy_with_same_intervals_merged()
    # %%
    tps = [ol for ol in overlaps if ol.text in ("c+c")]
    # %%
    # print([ol.text for ol in overlaps if ol.text != "c+c"])
    out_intervals = []
    tp_intervals = []
    fp_intervals = []
    TRUE_POSITIVE_LABELS = ("c+c", "c?+c")
    FALSE_POSITIVE_LABELS = ("no-c+c")
    # %%
    for ol in overlaps:
        if ol.text in TRUE_POSITIVE_LABELS:
            annot_creapy = creapy_tier.get_annotations_between_timepoints(
                ol.start_time, ol.end_time, left_overlap=True, right_overlap=True)[0]
            annot_own = own_tier.get_annotations_between_timepoints(
                ol.start_time, ol.end_time, left_overlap=True, right_overlap=True)[0]
            delta_start = annot_creapy.start_time - annot_own.start_time
            delta_end = annot_creapy.end_time - annot_own.end_time
            delta_length = delta_start - delta_end
            tp_intervals.append(CreakInterval(
                ol.start_time, ol.end_time, delta_start, delta_end, text="TP"))

        elif ol.text in FALSE_POSITIVE_LABELS:
            fp_intervals.append(ol)

    # %%
    # Delete overlaps
    for ol_interval in overlaps:
        own_tier_copy.delete_annotations_between_timepoints(
            ol_interval.start_time, ol_interval.end_time, left_overlap=True, right_overlap=True
        )
        creapy_tier_copy.delete_annotations_between_timepoints(
            ol_interval.start_time, ol_interval.end_time, left_overlap=True, right_overlap=True
        )
    # %%
    own_tier_creak_only = tgt.core.Tier(
        objects=own_tier_copy.get_annotations_with_text(pattern="c"))
    creapy_tier_creak_only = tgt.core.Tier(
        objects=creapy_tier_copy.get_annotations_with_text(pattern="c"))

    # %%
    eval_tier = tgt.core.IntervalTier(name=EVALUATION_TIER_NAME)
    for tp in tp_intervals:
        t0_start = tp.start_time - abs(tp.delta_start)
        t0_end = tp.end_time + abs(tp.delta_end)
        try:
            eval_tier.add_annotation(tp)
        except ValueError as e:
            eval_tier.delete_annotations_between_timepoints(
                tp.start_time, tp.end_time, left_overlap=True, right_overlap=True)
            try:
                eval_tier.add_annotation(tp)
            except ValueError as e:
                # print(e)
                pass
        try:
            eval_tier.add_annotation(tgt.core.Interval(
                start_time=t0_start, end_time=tp.start_time, text="+" if tp.delta_start > 0 else "-"))

        except ValueError as e:
            # print(e)
            pass

        try:
            eval_tier.add_annotation(tgt.core.Interval(
                start_time=tp.end_time, end_time=t0_end, text="+" if tp.delta_end < 0 else "-"))
        except ValueError as e:
            # print(e)
            pass
    # %%
    for fns in own_tier_creak_only.annotations:
        try:
            eval_tier.add_interval(tgt.core.Interval(
                fns.start_time, fns.end_time, text="FN"))
        except ValueError as e:
            print(e)

    for fps in creapy_tier_creak_only.annotations:
        try:
            eval_tier.add_interval(tgt.core.Interval(
                fps.start_time, fps.end_time, text="FP"))
        except ValueError as e:
            print(e)

    for fps in fp_intervals:
        try:
            eval_tier.add_interval(tgt.core.Interval(
                fps.start_time, fps.end_time, text="FP"))
        except ValueError as e:
            eval_tier.delete_annotations_between_timepoints(
                fps.start_time, fps.end_time, left_overlap=True, right_overlap=True)
            try:
                eval_tier.add_interval(tgt.core.Interval(
                    fps.start_time, fps.end_time, text="FP"))
            except ValueError as e:
                print(e)

    # %%

    TP = len(tp_intervals)
    FP = len(fp_intervals) + len(creapy_tier_creak_only)
    FN = len(own_tier_creak_only)

    F1 = TP / (TP + (FP + FN) / 2) if (TP + (FP + FN) / 2) != 0 else np.nan
    Precision = TP / (TP + FP) if (TP + FP) != 0 else np.nan
    Recall = TP / (TP + FN) if (TP + FP) != 0 else np.nan

    tg.add_tier(eval_tier)
    if tier_name_evaluation is not None:
        tgt.io.write_to_file(tg, TEXTGRID_PATH, encoding="utf-16")

    return F1, Precision, Recall, TP, FP, FN


def main():
    TEXTGRID_PATH = "/home/creaker/tip/stateofgrass/GRASS/004M024F/004M024F_HM2_HM1_CS_001_creak.TextGrid"
    OWN_TIER_NAME = "024F-creak"
    OUTFILE = "/home/creaker/tip/results/evaluation.txt"
    CREAPY_TIER_NAME = "024F-creapy-F"
    # INTERVAL_LABEL = "c"
    EVALUATION_TIER_NAME = "creapy_evaluation"

    # %%
    for encoding in ("utf-8", "utf-16"):
        try:
            tg = tgt.io.read_textgrid(TEXTGRID_PATH, encoding=encoding)
        except UnicodeDecodeError as e:
            print(f"Error occured reading textfile:\n\n{e}")
        else:
            break

    own_tier: tgt.core.Tier = tg.get_tier_by_name(OWN_TIER_NAME)
    creapy_tier: tgt.core.Tier = tg.get_tier_by_name(CREAPY_TIER_NAME)
    # Grausliche loesung im moment
    t_min = max(own_tier.intervals[0].start_time,
                creapy_tier.intervals[0].start_time)
    t_max = min(own_tier.intervals[-1].end_time,
                creapy_tier.intervals[-1].end_time)

    own_tier = tgt.core.IntervalTier(objects=own_tier.get_annotations_between_timepoints(
        start=t_min, end=t_max, left_overlap=True, right_overlap=True))
    creapy_tier = tgt.core.IntervalTier(objects=creapy_tier.get_annotations_between_timepoints(
        start=t_min, end=t_max, left_overlap=True, right_overlap=True))
    overlaps = tgt.util.get_overlapping_intervals(
        own_tier,
        creapy_tier,
    )
    eval_tier = tgt.core.IntervalTier(name="creapy-evaluation")
    own_tier_copy = own_tier.get_copy_with_same_intervals_merged()
    creapy_tier_copy = creapy_tier.get_copy_with_same_intervals_merged()
    # %%
    tps = [ol for ol in overlaps if ol.text in ("c+c")]
    # %%
    print([ol.text for ol in overlaps if ol.text != "c+c"])
    out_intervals = []
    tp_intervals = []
    fp_intervals = []
    TRUE_POSITIVE_LABELS = ("c+c", "c?+c")
    FALSE_POSITIVE_LABELS = ("no-c+c")
    # %%
    for ol in overlaps:
        if ol.text in TRUE_POSITIVE_LABELS:
            annot_creapy = creapy_tier.get_annotations_between_timepoints(
                ol.start_time, ol.end_time, left_overlap=True, right_overlap=True)[0]
            annot_own = own_tier.get_annotations_between_timepoints(
                ol.start_time, ol.end_time, left_overlap=True, right_overlap=True)[0]
            delta_start = annot_creapy.start_time - annot_own.start_time
            delta_end = annot_creapy.end_time - annot_own.end_time
            delta_length = delta_start - delta_end
            tp_intervals.append(CreakInterval(
                ol.start_time, ol.end_time, delta_start, delta_end, text="TP"))

        elif ol.text in FALSE_POSITIVE_LABELS:
            fp_intervals.append(ol)

    # %%
    # Delete overlaps
    for ol_interval in overlaps:
        own_tier_copy.delete_annotations_between_timepoints(
            ol_interval.start_time, ol_interval.end_time, left_overlap=True, right_overlap=True
        )
        creapy_tier_copy.delete_annotations_between_timepoints(
            ol_interval.start_time, ol_interval.end_time, left_overlap=True, right_overlap=True
        )
    # %%
    own_tier_creak_only = tgt.core.Tier(
        objects=own_tier_copy.get_annotations_with_text(pattern="c"))
    creapy_tier_creak_only = tgt.core.Tier(
        objects=creapy_tier_copy.get_annotations_with_text(pattern="c"))

    # %%
    eval_tier = tgt.core.IntervalTier(name="creapy-evaluation")
    for tp in tp_intervals:
        t0_start = tp.start_time - abs(tp.delta_start)
        t0_end = tp.end_time + abs(tp.delta_end)
        try:
            eval_tier.add_annotation(tp)
        except ValueError as e:
            eval_tier.delete_annotations_between_timepoints(
                tp.start_time, tp.end_time, left_overlap=True, right_overlap=True)
            try:
                eval_tier.add_annotation(tp)
            except ValueError as e:
                print(e)
        try:
            eval_tier.add_annotation(tgt.core.Interval(
                start_time=t0_start, end_time=tp.start_time, text="+" if tp.delta_start > 0 else "-"))

        except ValueError as e:
            print(e)
            pass

        try:
            eval_tier.add_annotation(tgt.core.Interval(
                start_time=tp.end_time, end_time=t0_end, text="+" if tp.delta_end < 0 else "-"))
        except ValueError as e:
            print(e)
            pass
    # %%
    for fns in own_tier_creak_only.annotations:
        try:
            eval_tier.add_interval(tgt.core.Interval(
                fns.start_time, fns.end_time, text="FN"))
        except ValueError as e:
            print(e)

    for fps in creapy_tier_creak_only.annotations:
        try:
            eval_tier.add_interval(tgt.core.Interval(
                fps.start_time, fps.end_time, text="FP"))
        except ValueError as e:
            print(e)

    for fps in fp_intervals:
        try:
            eval_tier.add_interval(tgt.core.Interval(
                fps.start_time, fps.end_time, text="FP"))
        except ValueError as e:
            eval_tier.delete_annotations_between_timepoints(
                fps.start_time, fps.end_time, left_overlap=True, right_overlap=True)
            try:
                eval_tier.add_interval(tgt.core.Interval(
                    fps.start_time, fps.end_time, text="FP"))
            except ValueError as e:
                print(e)

    # %%

    TP = len(tp_intervals)
    FP = len(fp_intervals) + len(creapy_tier_creak_only)
    FN = len(own_tier_creak_only)
    F1 = TP / (TP + (FP + FN) / 2)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    # %%
    tg.add_tier(eval_tier)
    # tgt.io.write_to_file(tg, TEXTGRID_PATH, encoding="utf-16")
    # %%
    own_tier_creak_only
    # %%
    print(
        f"""True Positives (TP):\t{TP}
    False Positives (FP):\t{FP}
    False Negatives (FN):\t{FN}
    F1-Score: {F1:.3f}
    Precision: {Precision: .3f}
    Recall: {Recall: .3f}
    """
    )

    # %%
    config = get_config()
    _ste = config["MODEL"]["POSTPROCESSING"]["NON_MODAL_EXCLUSION"]["VALUES"]["STE"]["threshold"]
    _zcr = config["MODEL"]["POSTPROCESSING"]["NON_MODAL_EXCLUSION"]["VALUES"]["ZCR"]["threshold"]
    _block_size = config["MODEL"]["PREPROCESSING"]["block_size"]
    _hop_size = config["MODEL"]["PREPROCESSING"]["hop_size"]
    _intervals = config["MODEL"]["POSTPROCESSING"]["INTERVALS"]

    # %%

    content = f"""{datetime.now().strftime(" [%d/%m/%Y, %H:%M:%S] ").center(120, "-")}

    PRAAT STUFF:
        Textgrid File: {TEXTGRID_PATH}
        Own Tier: {OWN_TIER_NAME}
        Creapy Tier: {CREAPY_TIER_NAME}

    THRESHOLDS:
        STE: {_ste}
        ZCR: {_zcr}
        creak_threshold: {_intervals["creak_threshold"]}

    GAPS:
        min_creak_length: {_intervals["min_creak_length"]}
        max_gap: {_intervals["max_gap"]}

    EVALUATION METRICS:
        F1: {F1:.3f}
        Precision: {Precision: .3f}
        Recall: {Recall: .3f}
        True Positives: {TP}
        False Positives: {FP}
        False Negatives: {FN}

    """
    with open(OUTFILE, "a") as f:
        f.write(content)
        f.write("-" * 120 + '\n')


if __name__ == "__main__":
    main()
