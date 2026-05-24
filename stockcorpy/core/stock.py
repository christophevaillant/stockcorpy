import os
from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from enum import Enum

from massive import RESTClient

from .data import Data, RawDataPoint, DataPointError, ProcessedDataPoint

logger = logging.getLogger("stock")

class StockError(DataPointError):
    """Error specific to the stock class"""
    pass


class StockNotFoundError(DataPointError):
    """Error specific to the price classes"""
    pass

class StockProcessingMode(Enum):
    VALUE = 0
    DIFFERENCE = 1


class Stock(Data):
    """Derived from the Data class, this class implements the specific functions
    for stock prices, using the massive api."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def create_data(self, number_of_days: int):
        """Download the specific stock's price history from the polygon source"""

        client = RESTClient(os.environ["POLYGON_API_KEY"])
        now = datetime.now()
        timespan = now - timedelta(days=number_of_days)
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
        super().create_data(number_of_days)

    def process_data(
            self,
            mode: StockProcessingMode = StockProcessingMode.VALUE,
            offset_days = 1
        ):
        super().process_data(offset_days=offset_days)
        if mode == StockProcessingMode.DIFFERENCE:
            days = [point.days for point in self.processed_data]
            values = [point.value for point in self.processed_data]
            new_data = []
            for i in range(1, len(self.processed_data)):
                if days[i-1] == days[i] - 1:
                    new_data.append(
                        ProcessedDataPoint(
                            days=days[i],
                            value= values[i] - values[i-1]
                            )
                        )
            self.processed_data = new_data
    
    def plot_data(self, graph_file: Path | None = None):
        super().plot_data("Stock price", graph_file=graph_file)
