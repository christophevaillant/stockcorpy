import os
from datetime import date, datetime, timedelta
import logging

from massive import RESTClient

from .data import Data, RawDataPoint,DataPointError

logger = logging.getLogger("stock")

class StockError(DataPointError):
    """Error specific to the stock class"""
    pass


class StockNotFoundError(DataPointError):
    """Error specific to the price classes"""
    pass


class Stock(Data):
    """Derived from the Data class, this class implements the specific functions
    for stock prices, using the massive api."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def create_data(self, number_of_days: int):
        """Download the specific stock's price history from the polygon source"""

        client = RESTClient(os.environ("POLYGON_API_KEY"))
        now = datetime.datetime.now()
        timespan = now - datetime.timedelta(days=number_of_days)
        ticker = client.list_aggs(
            self.name,
            1, 
            "day", 
            timespan.strftime("%Y-%m-%d"),
            now.strftime("%Y-%m-%d"), limit=50000
        )
        existing_dates = self.retrieve_dates()
        for datum in ticker:
            ticker_date = date.fromtimestamp(datum.timestamp / 1000)
            if ticker_date not in existing_dates:
                self.raw_data.append(RawDataPoint(
                    date=ticker_date,
                    value=datum.open))

    def process_data(self, offset_days = -1):
        return super().process_data(offset_days=offset_days)
    
    def plot_data(self, graph_file: Path | None = None):
        super().plot_data("Stock price", graph_file=graph_file)
