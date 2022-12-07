import numpy as np
from abc import ABC, abstractmethod
import os
from stockcorpy.utils import StockcorError, WriteJSON, LoadJSON
import datetime
import time
import logging
import pandas as pd
import pylab as pl
import copy


class PriceError(StockcorError):
    """Error specific to the price classes"""
    pass


class PriceNotFoundError(StockcorError):
    """Error specific to the price classes"""
    pass


class Price(ABC):
    """Base class for a generic stock or coin. Will need overwriting
    with specific methods that are able to call the relevant
    data sources."""

    def __init__(self, name, time_average=7):
        self.name = name
        self.priceDir = os.path.join(os.getcwd(), name)

        # Data and its corresponding filename
        self.raw_data = None
        self.data_filename = os.path.join(self.priceDir,
                                          f"{self.name}_raw.csv")

        # Define a few different time series and numbers that will be needed
        self.time = None
        self.avg_data = None
        self.clean_data = None
        self.noise_data = None
        self.stdev = 1.0

        # Configs and their corresponding filename
        self.configs = {
            "time_average": time_average,
            "initial_time": 0,
            "final_time": 0,
        }
        self.config_filename = os.path.join(self.priceDir,
                                            f"{self.name}_configs.json")

        # Model-specific items
        self.models = {"correlated": [], "predictions": {}}
        self.models_filename = os.path.join(self.priceDir,
                                            f"{self.name}_models.json")
        self.nn_integrator = None

    @abstractmethod
    def CreatePrice(self):
        """Download the specific price from the data source and
        grab the history. Also set up the directory and save
        the raw data."""

        self.raw_data.sort_values("time")

        # Check the directory exists, otherwise create it
        if not os.path.isdir(self.priceDir):
            os.mkdir(self.priceDir)

        # Save the raw data if it has been loaded
        self.SavePrice()
        return None

    @abstractmethod
    def UpdatePrice(self):
        """Download any data points that may have occured since the last time
        the data source was accessed."""

        return None

    def LoadPrice(self):
        """Load the raw data and configs from a specified file."""

        # First load data
        if os.path.exists(self.data_filename):
            self.raw_data = pd.read_csv(self.data_filename, index_col=0)
            self.raw_data["time"] = self.raw_data["raw_time"].copy()
            self.raw_data.sort_values("time")
        else:
            raise PriceError(f"Could not find data file {self.data_filename}")

        # Now load configs
        if os.path.exists(self.config_filename):
            self.configs = LoadJSON(self.config_filename)
        else:
            raise PriceError(f"Could not find configs file {self.config_filename}")

        return None

    def SavePrice(self, backup=False):
        """Save the current price to the relevant file."""

        if backup:
            os.rename(self.config_filename, f"{self.config_filename}.backup")
            os.rename(self.models_filename, f"{self.models_filename}.backup")
            os.rename(self.data_filename, f"{self.data_filename}.backup")

        if self.raw_data is not None:
            self.raw_data.to_csv(self.data_filename, columns=["raw_time", self.name])

        WriteJSON(self.config_filename, self.configs)

        models_output = copy.deepcopy(self.models)
        for date in self.models["predictions"].keys():
            models_output["predictions"][date] = self.models["predictions"][date]["average"].tolist()
        WriteJSON(self.models_filename, models_output)

        return None

    def ProcessPrice(self):
        """Create a few of the time series from the raw data in order to make
        these available for other methods"""

        from stockcorpy.utils import MovingAverage

        if self.clean_data is not None:
            self.avg_data = pd.Series(MovingAverage(self.clean_data[self.name].values,
                                                    self.configs["time_average"]))
            self.noise_data = pd.Series(np.gradient(self.clean_data[self.name].values))
            self.stdev = np.std(self.noise_data)
            self.noise_data.to_csv(f"{self.name}_noise.csv")
            print(f"Coin {self.name} has saved noise data from clean")
        elif self.raw_data is not None:
            self.avg_data = MovingAverage(self.raw_data, self.configs["time_average"])
            self.noise_data = self.raw_data[self.configs["time_average"]-1:] - self.avg_data
            self.stdev = np.std(self.noise_data)
            self.noise_data.to_csv(f"{self.name}_noise.csv")
            print(f"Coin {self.name} has saved noise data from raw")
        else:
            raise PriceError("No data currently loaded.")
        return None

    def Graph(self, start_time=-168):
        """Plot the various properties of the coin"""

        if self.raw_data is not None:
            pl.plot(self.raw_data["time"].values, self.raw_data[self.name].values)
        # if self.avg_data is not None:
        #     pl.plot(self.avg_data.values[:, 0], self.avg_data.values[:, 1])
        if len(self.models["predictions"]) > 0:
            for i, date in enumerate(self.models["predictions"].keys(), start=1):
                for trial in self.models["predictions"][date]:
                    pl.plot(self.models["predictions"][date][trial][:, 0],
                            self.models["predictions"][date][trial][:, 1],
                            color='k', alpha=0.1)
                pl.plot(self.models["predictions"][date]["average"][:, 0],
                            self.models["predictions"][date]["average"][:, 1])
        else:
            pl.plot(self.raw_data.values[:, 0], self.raw_data.values[:, 1])
        pl.gca().set_xlim(left=self.raw_data["time"].values[start_time])
        pl.ylabel(self.name)
        pl.show()


class Coin(Price):
    """Derived from the Price class, this class implements the specific functions
    for Crypto currencies, using the CoinGecko python API. The choice of forcing
    the name of the coin to be the same as the coin code is deliberate to avoid
    the possibility of creating inconsistencies later."""

    def __init__(self, name, time_average=7):
        super().__init__(name, time_average=time_average)

    def CreatePrice(self):
        """Download the specific coin's price history from the CoinGecko source"""
        from pycoingecko import CoinGeckoAPI

        # ##########
        # Specific coingecko retrieval
        # ##########
        cg = CoinGeckoAPI()

        # overwrite the initial time
        self.configs["initial_time"] = (datetime.datetime.now() -
                                        datetime.timedelta(hours=24)).timestamp()
        # Retrieve the data
        try:
            self.raw_data = pd.DataFrame(cg.get_coin_market_chart_range_by_id(
                self.name, "eur", self.configs["initial_time"],
                datetime.datetime.now().timestamp())['prices'],
                                         columns=["raw_time", self.name])
            self.raw_data["time"] = self.raw_data["raw_time"].copy()
        except ValueError:
            raise PriceError(f"Could not find coin with id {self.name}")
            # Set the final time
        if len(self.raw_data.values) == 0:
            raise PriceNotFoundError(f"Could not find coin {self.name}.")
        print(f"Downloaded {self.name}")
        # Unix time in ms
        self.configs["final_time"] = datetime.datetime.now().timestamp()

        # Set the zero, time is returned in milliseconds
        # self.raw_data["time"] -= self.configs["initial_time"] * 1000.0

        # ##########
        # Do all the standard stuff
        # ##########
        super().CreatePrice()

        return None

    def UpdatePrice(self):
        """Download any data that may have been created since the last download."""
        from pycoingecko import CoinGeckoAPI

        cg = CoinGeckoAPI()
        # Download the data
        new_data = pd.DataFrame(cg.get_coin_market_chart_range_by_id(
            self.name, "eur", self.configs["final_time"],
            datetime.datetime.now().timestamp())['prices'],
                                columns=["raw_time", self.name])
        print(new_data)
        if len(new_data) == 0:
            logging.warning(f"Nothing to be updated for coin {self.name}.")
        else:
            self.raw_data = pd.concat([self.raw_data, new_data], ignore_index=True)
            self.raw_data["time"] = self.raw_data["raw_time"].copy()
            # self.raw_data["time"] -= self.configs["initial_time"] * 1000.0
            self.configs["final_time"] = datetime.datetime.now().timestamp()
            self.SavePrice(backup=True)

        return None
