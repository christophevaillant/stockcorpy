import numpy as np
import pandas as pd
import logging
from functools import reduce
from copy import deepcopy
from stockcorpy.utils import StockcorError
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime


class MethodsError(StockcorError):
    """Error specific to the methods functions"""
    pass


def RemoveMissingData(coin_list, time_unit=3600000.0):
    """Function that removes the entries where any points are missing, and sets
    the time scale (default time unit is hours), with the zero chosen as the
    earliest time in the seris."""

    # Find the earliest time and change the time stamps to be in the right units
    time_zero_list = []
    invalids = []
    for coin in coin_list.values():
        try:
            time_zero_list.append(coin.raw_data["time"].iat[0])
        except (TypeError, KeyError):
            invalids.append(coin.name)
            continue
    time_zero = min(time_zero_list)

    for coin in invalids:
        logging.warning(f"Coin {coin.name} has length 0, removing...")
        coin_list.pop(coin)

    for coin in coin_list.values():
        coin.raw_data["time"] = np.round((coin.raw_data["time"] - time_zero)/time_unit)

    # Make a pandas data frame for ease
    coin_df = pd.DataFrame(next(iter(coin_list.values())).raw_data["time"])
    for coin in coin_list.values():
        coin.raw_data.drop_duplicates(subset="time", keep="last", inplace=True)
        coin_df = coin_df.merge(coin.raw_data[["time", coin.name]], on="time", how="left", copy=False)
        coin_df.drop_duplicates(subset="time", keep="last", inplace=True)

    # Clean the data by removing columns that have holes
    coin_df.fillna(axis=0, inplace=True, method="ffill")
    print(coin_df)
    for coin in coin_list.values():
        coin.clean_data = coin_df[["time", coin.name]]

    return coin_df


def CalculateCorrelations(coin1, coin2, length=70):
    """Calculate the correlation function between the noise of two coins, and work out
    the cross-correlation time and the pearson correlation coefficient."""

    logging.debug(f"Calculating correlation between coins {coin1.name} and {coin2.name}")
    # First, a quick check that the two data sets are cleaned and that they're
    # the same length
    if len(coin1.noise_data) != len(coin2.noise_data):
        raise MethodsError(f"Coins {coin1.name} and {coin2.name} are not the same length "
                           f"({len(coin1.noise_data)} and {len(coin2.noise_data)}).")
    else:
        noise_length = len(coin1.noise_data)

    window = int(np.floor(noise_length/length))
    correlation = np.zeros((length))

    logging.debug(f"Using window length {length} and {window} windows. Total length is {noise_length}")

    coin1_reshaped = np.reshape(coin1.noise_data.values[0: window*length], (window, length))
    coin2_reshaped = np.reshape(coin2.noise_data.values[0: window*length], (window, length))
    correlation = np.zeros((length), dtype='float64')
    for i in range(window):
        for j in range(length):
            correlation[j] += coin1_reshaped[i, 0] * coin2_reshaped[i, j]
    
    correlation /= length
    corr_time = np.trapz(correlation)/np.sqrt(np.square(coin1.stdev) * np.square(coin2.stdev))

    pearson = np.corrcoef(coin1.grad_data[coin1.configs["time_average"]-1:], y=coin2.noise_data)

    return (corr_time, pearson[0,1])


def TrainNetworks(wallet):
    """Go through each coin, load the correlated coins, and normalize the data. Using that,
    train a neural network for each coin."""

    # Blacklist to avoid coins that are dodgy
    blacklist = []

    for parent in wallet.values():
        n_cor = len(parent.models["correlated"])
        if  n_cor == 0:
            print(f"Coin {parent.name} has no correlations, adding to blacklist")
            blacklist.append(parent.name)
            continue

        # Set up the NN object
        print(f"Fitting {parent.name} with {parent.models['correlated']}")
        # nn = MLPRegressor(solver="lbfgs", hidden_layer_sizes=(max(1,round(0.5*(n_cor+1))),),
        #                   alpha=0.0, max_iter=100000)
        nn = LinearRegression()
        x = np.zeros((len(parent.noise_data), n_cor), dtype='float')

        for i, child in enumerate(parent.models["correlated"]):
            x[:, i] = (wallet[child].grad_data[wallet[child].configs["time_average"]-2:-1] /
                       wallet[child].grad_stdev)

        nn.fit(x, parent.noise_data[1:] / parent.stdev)
        # print(f"Coin {parent.name} ran with {nn.n_iter_} iterations")

        parent.nn_integrator = nn

    for coin in blacklist:
        del wallet[coin]

    print(wallet.keys())

    return wallet


def PredictCoins(wallet, n_steps, n_trials=10):
    """Take all the coins in a coin dict and step forward n_steps times (in the chosen units).
    This will update the coins' prediction dicts for plotting."""

    x = {}
    date = datetime.now().strftime("%d-%m-%Y")
    for coin in wallet.values():
        coin.models["predictions"][date] = {}
    for trial in range(n_trials):
        for coin in wallet.values():
            coin.models["predictions"][date][trial] = np.zeros((n_steps, 2))
            x[coin.name] = np.zeros((1, len(coin.models["correlated"])))
        for i in range(n_steps):
            for coin_name, coin in wallet.items():
                wallet[coin_name] = TakeStep(coin, i, x, wallet, date, trial)

    for coin in wallet.values():
        coin.models["predictions"][date]["average"] = np.copy(coin.models["predictions"][date][0])
        coin.models["predictions"][date]["average"][:, 1] = \
            np.average([val[:, 1] for val in coin.models["predictions"][date].values()], axis=0)
    
    return None


def TakeStep(coin, step, x, wallet, date, trial):
    for i, child in enumerate(coin.models["correlated"]):
        coin.models["predictions"][date][trial][step, 0] = coin.raw_data["time"].iat[-1] + float(step)

    if step == 0:
        for i, child in enumerate(coin.models["correlated"]):
            x[coin.name][0, i] = wallet[child].grad_data.iat[-1] / wallet[child].grad_stdev
        x[coin.name].reshape(1, -1)
        delta = (coin.nn_integrator.predict(x[coin.name])) * coin.stdev
        # delta = (np.random.normal() + coin.nn_integrator.predict(x[coin.name])) * coin.stdev
        coin.models["predictions"][date][trial][0, 1] = (coin.raw_data[coin.name].iat[-1] + delta)
    elif step == 1:
        for i, child in enumerate(coin.models["correlated"]):
            x[coin.name][0, i] = (wallet[child].models["predictions"][date][trial][step-1, 1] -
                                  wallet[child].clean_data[child].iat[-1]) / wallet[child].stdev
        x[coin.name].reshape(1, -1)
        delta = (coin.nn_integrator.predict(x[coin.name])) * coin.stdev
        # delta = (np.random.normal() + coin.nn_integrator.predict(x[coin.name])) * coin.stdev
        # print(coin.name, coin.models["predictions"][date][trial][step-1], delta)
        coin.models["predictions"][date][trial][step, 1] = (coin.models["predictions"][date][trial][step-1, 1] +
                                                            delta)
    else:
        for i, child in enumerate(coin.models["correlated"]):
            x[coin.name][0, i] = (wallet[child].models["predictions"][date][trial][step-1, 1] -
                                  wallet[child].models["predictions"][date][trial][step-2, 1]) / wallet[child].stdev
        x[coin.name].reshape(1, -1)
        delta = (coin.nn_integrator.predict(x[coin.name])) * coin.stdev
        # delta = (np.random.normal() + coin.nn_integrator.predict(x[coin.name])) * coin.stdev
        # print(coin.name, coin.models["predictions"][date][trial][step-1], delta)
        coin.models["predictions"][date][trial][step, 1] = (coin.models["predictions"][date][trial][step-1, 1] +
                                                     delta)
            
    return coin

