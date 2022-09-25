import numpy as np
from stockcorpy.utils import MovingAverage
import pandas as pd

def RemoveMissingData(coin_list, time_unit=3600000.0):
    """Function that removes the entries where any points are missing, and sets
    the time scale (default time unit is hours), with the zero chosen as the
    earliest time in the seris."""

    # Find the earliest time and change the time stamps to be in the right units
    time_zero = min([coin.raw_data[0,0] for coin in coin_list])
    coin_list[:][:, 0] = np.round((coin_list[:][:, 0] - time_zero)/num_hours)

    # Clean the data by removing columns that have holes
    for coin in coin_list:
        coin.clean_data = np.array([x for x in doge if x[0] in bitc[:, 0]])
    doge_clean = 
    bitc_clean = np.array([x for x in bitc if x[0] in doge[:, 0]])
    print(np.shape(doge_clean), np.shape(bitc_clean))
    timestamp = doge_clean[:, 0]
