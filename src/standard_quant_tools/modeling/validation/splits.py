"""Simple time-ordered holdout split — for small datasets/tests where a
full walk-forward split has too few dates to produce even one fold.
ValidationSpec.method is currently a Literal["walk_forward"] only; this
file is the natural place a future "holdout" method would live."""

from typing import Tuple

import numpy as np
import pandas as pd

from standard_quant_tools.error import ValidationError


def holdout_split(
    dates: pd.Index, train_frac: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < train_frac < 1.0):
        raise ValidationError(f"train_frac must be in (0, 1), got {train_frac}")
    n = len(dates)
    split_at = int(n * train_frac)
    if split_at == 0 or split_at == n:
        raise ValidationError(
            f"holdout_split: train_frac={train_frac} leaves an empty fold for n={n} dates"
        )
    return np.arange(0, split_at), np.arange(split_at, n)
