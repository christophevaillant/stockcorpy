import numpy as np
from abc import ABC
import os
from stockcorpy.utils import StockcorError, WriteJSON, LoadJSON
import datetime
import time
import logging

class PriceError(StockcorError):
    """Error specific to the price classes"""
    pass


class Price(ABC):
    """Base class for a generic stock or coin. Will need overwriting
    with specific methods that are able to call the relevant
    data sources."""

    def __init__(self, name, time_average=7):
        self.name = name
        self.priceDir=os.path.join(os.getcwd(), name)

        # Data and its corresponding filename
        self.raw_data = None
        self.data_filename = os.path.join(self.priceDir, f"{self.name}_raw.csv")

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
        self.config_filename = os.path.join(self.priceDir, f"{self.name}_configs.json")

    @abstractmethod
    def CreatePrice(self):
        """Download the specific price from the data source and
        grab the history. Also set up the directory and save
        the raw data."""

        # Check the directory exists, otherwise create it
        if not os.isdir(self.priceDir):
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
            self.raw_data = np.loadtxt(self.data_filename)
        else:
            raise PriceError(f"Could not find data file {self.data_filename}")

        # Now load configs
        if os.path.exists(self.config_filename):
            self.configs = LoadJSON(self.config_filename)
        else:
            raise PriceError(f"Could not find configs file {self.config_filename}")

        return None

    def SavePrice(self):
        """Save the current price to the relevant file."""

        if self.raw_data is not None:
            np.savetxt(self.data_filename, self.raw_data)

        WriteJSON(self.config_filename, self.configs)

        return None

    def ProcessPrice(self):
        """Create a few of the time series from the raw data in order to make
        these available for other methods"""

        from stockcorpy.utils import MovingAverage

        if self.clean_data is not None and not reprocess:
            self.avg_data = MovingAverage(self.clean_data, self.configs["time_average"])
            self.noise_data = self.clean_data[self.configs["time_average"]-1:] - self.avg_data
            self.stdev = np.std(self.noise_data)
        elif self.raw_data is not None:
            self.avg_data = MovingAverage(self.raw_data, self.configs["time_average"])
            self.noise_data = self.raw_data[self.configs["time_average"]-1:] - self.avg_data
            self.stdev = np.std(self.noise_data)
        else:
            raise PriceError("No data currently loaded.")


class Coin(Price):
    """Derived from the Price class, this class implements the specific functions
    for Crypto currencies, using the CoinGecko python API. The choice of forcing
    the name of the coin to be the same as the coin code is deliberate to avoid
    the possibility of creating inconsistencies later."""

    def __init__(self, name, time_average=7):
        super().__init__(self, name, time_average=7)


    def CreatePrice(self):
        """Download the specific coin's price history from the CoinGecko source"""

        # ##########
        # Specific coingecko retrieval
        # ##########

        # overwrite the initial time
        self.configs["initial_time"] = time.mktime(datetime.now().timetuple() -
                                                   datetime.timedelta(days=89))
        # Retrive the data
        self.raw_data = np.array(cg.get_coin_market_chart_range_by_id(
            self.name, "eur", self.configs["initial_time"],
            time.mktime(datetime.now().timetuple()))['prices'])
        # Set the final time
        self.configs["final_time"] = self.raw_data[-1, 0]
        self.time = self.raw_data[:, 0] - self.configs["final_time"]

        # ##########        
        # Do all the standard stuff
        # ##########
        super().CreatePrice()

        return None

    def UpdatePrice(self):
        """Download any data that may have been created since the last download."""

        # Download the data
        new_data = np.array(cg.get_coin_market_chart_range_by_id(
            self.name, "eur", self.configs["final_time"],
            time.mktime(datetime.now().timetuple()))['prices'])

        if new_data[-1, 0] > self.configs["final_time"]:
            self.raw_data = np.append(self.raw_data, new_data)
        else:
            logging.warning(f"Nothing to be updated for coin {self.name}.")

        return None
