import json
import os
import numpy as np


class StockcorError(Exception):
    """Generic class for error reporting"""
    pass


def LoadJSON(input_path):
    """Function to load a json with standard settings."""

    if os.path.exists(input_path):
        with open(input_path, "r") as open_file:
            imported_json = json.load(open_file)
    else:
        raise StockcorError(f"Could not find file {input_path}")

    return imported_json


def WriteJSON(input_path, input_dict):
    """Function to write a dict to file as json."""

    try:
        with open(input_path, "w") as open_file:
            json.dump(input_dict, open_file, ensure_ascii=False, indent=4, default=convert)
    except:
        raise StockcorError(f"Could not write to file {input_path} with json {input_dict} and "
                            f"type {type(input_dict)}")

    return None


def MovingAverage(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError
