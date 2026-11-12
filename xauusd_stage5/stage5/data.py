import numpy as np
import pandas as pd

from . import s5config as S


def load_m15(usecols=None):
    d = pd.read_csv(S.M15_MTF, usecols=usecols)
    d["time"] = pd.to_datetime(d["time"])
    return d


def load_m5():
    d = pd.read_csv(S.M5_NORM)
    d["time"] = pd.to_datetime(d["time"])
    return d.reset_index(drop=True)
