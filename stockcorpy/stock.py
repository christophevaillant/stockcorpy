import numpy as np
from abc import ABC
import os
from stockcorpy.utils import StockcorError, WriteJSON, LoadJSON

class StockError(StockcorError):
    """Error specific to the stock classes"""
    pass


class Stock(ABC):
    """Base class for a generic stock or coin. Will need overwriting
    with specific methods that are able to call the relevant
    data sources."""

    def __init__(self, name):
        self.name = name
        self.stockDir=os.path.join(os.getcwd(), name)

        # Data and its corresponding filename
        self.raw_data = None
        self.data_filename = os.path.join(self.stockDir, f"{self.name}_raw.csv")

        # Configs and their corresponding filename
        self.configs = {}
        self.config_filename = os.path.join(self.stockDir, f"{self.name}_configs.json")

    @abstractmethod
    def CreateStock(self):
        """Download the specific stock from the data source and
        grab the history. Also set up the directory and save
        the raw data."""

        # Check the directory exists, otherwise create it
        if not os.isdir(self.stockDir):
            os.mkdir(self.stockDir)

        # Save the raw data if it has been loaded
        if self.raw_data is not None:
            np.savetxt(self.data_filename, self.raw_data)

        WriteJSON(self.config_filename, self.configs)

        return None

    @abstractmethod
    def LoadStock(self):
        """Load the raw data and configs from a specified file."""

        # First load data
        if os.path.exists(self.data_filename):
            self.raw_data = np.loadtxt(self.data_filename)
        else:
            raise StockError(f"Could not find data file {self.data_filename}")

        # Now load configs
        if os.path.exists(self.config_filename):
            self.configs = LoadJSON(self.config_filename)
        else:
            raise StockError(f"Could not find configs file {self.config_filename}")

        return None
