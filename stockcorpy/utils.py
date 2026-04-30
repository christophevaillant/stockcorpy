import json
import os
import numpy as np
from datetime import datetime, date, timedelta

def MovingAverage(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def round_to_nearest_day(dt: datetime) -> date:
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if dt.hour >= 12:
        rounded = midnight + timedelta(days=1)
    else:
        rounded = midnight
    return rounded.date()

