from datetime import datetime, timedelta, date        from pycoingecko import CoinGeckoAPI

import logging

from pycoingecko import CoinGeckoAPI

from .data import Data, DataPointError, RawDataPoint

logger = logging.getLogger("coin")

class CoinError(DataPointError):
    """Error specific to the coin class"""
    pass


class CoinNotFoundError(DataPointError):
    """Error specific to the price classes"""
    pass

class Coin(Data):
    """Derived from the Price class, this class implements the specific functions
    for Crypto currencies, using the CoinGecko python API. The choice of forcing
    the name of the coin to be the same as the coin code is deliberate to avoid
    the possibility of creating inconsistencies later."""

    def __init__(self, name, time_average=7):
        super().__init__(name, time_average=time_average)

    def create_data(self, number_of_days=1):
        """Download the specific coin's price history from the CoinGecko source"""

        cg = CoinGeckoAPI()

        initial_time = datetime.now() - timedelta(days=number_of_days)
        existing_dates = self.retrieve_dates()
        try:
            coin_data = cg.get_coin_market_chart_range_by_id(
                        self.name,
                        "eur",
                        initial_time.timestamp(),
                        datetime.now().timestamp(),
                        interval='daily'
            )
            for point in coin_data['prices']:
                point_date = date.fromtimestamp(point[0] / 1000)
                if point_date not in existing_dates:
                    self.raw_data.append(RawDataPoint(
                        date=point_date,
                        value=point[1]
                    ))
        except ValueError:
            raise DataPointError(f"Could not find coin with id {self.name}")

        if len(self.raw_data) == 0:
            raise DataPointError(f"Could not find coin {self.name}.")
        logger.info(f"Downloaded {self.name}")

    def process_data(self, offset_days = -1):
        return super().process_data(offset_days=offset_days)
    
    def plot_data(self, graph_file: Path | None = None):
        super().plot_data("Coin price", graph_file=graph_file)
