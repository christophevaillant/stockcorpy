from .data import Data, DataPointError

class CoinError(DataPointError):
    """Error specific to the coin class"""
    pass


class CoinNotFoundError(DataPointError):
    """Error specific to the price classes"""
    pass

class Coin(Price):
    """Derived from the Price class, this class implements the specific functions
    for Crypto currencies, using the CoinGecko python API. The choice of forcing
    the name of the coin to be the same as the coin code is deliberate to avoid
    the possibility of creating inconsistencies later."""

    def __init__(self, name, time_average=7):
        super().__init__(name, time_average=time_average)

    def CreatePrice(self, days=1):
        """Download the specific coin's price history from the CoinGecko source"""
        from pycoingecko import CoinGeckoAPI

        # ##########
        # Specific coingecko retrieval
        # ##########
        cg = CoinGeckoAPI()

        # overwrite the initial time
        self.configs["initial_time"] = (datetime.datetime.now() -
                                        datetime.timedelta(days=days)).timestamp()
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

        # Set the zero, time is returned in milliseconds
        # self.raw_data["time"] -= self.configs["initial_time"] * 1000.0

        # ##########
        # Do all the standard stuff
        # ##########
        super().CreatePrice()

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
