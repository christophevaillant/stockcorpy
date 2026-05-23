from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from pickle import dump, load
from pathlib import Path
import logging

import pylab as pl

logger = logging.getLogger("data")

class DataPointError(RuntimeError):
    pass

@dataclass
class RawDataPoint:
    date: date
    value: float

@dataclass
class ProcessedDataPoint:
    days: float
    value: float

class Data(ABC):
    def __init__(self):
        self.raw_data: list[RawDataPoint] = []
        self.processed_data: list[ProcessedDataPoint] | None = None
        self.start_date: date = date.today()
        self.end_date: date = date.today()

    @staticmethod
    def load_from_file(filepath: Path):
        with open(filepath, "rb") as datafile:
             return load(datafile)

    def save_to_file(self, filepath: Path):
        with open(filepath, "wb") as datafile:
            dump(self, datafile)

    @abstractmethod
    def create_data(self, number_of_days: int):
        dates = [point.date for point in self.raw_data]
        if not dates:
            raise DataPointError("No dates found")
        self.start_date = min(dates)
        self.end_date = max(dates)
        length = self.end_date - self.start_date
        logger.info(f"Found data points between {self.start_date} and {self.end_date}, a total of {length.days} days")
        

    @abstractmethod
    def process_data(self, offset_days: int = 0):
        # self._pad_data()
        self._sort_data(offset_days)

    def _pad_data(self):
        number_days = self.end_date - self.start_date
        logger.debug(f"Found number of days {number_days.days} between {self.start_date} and {self.end_date}")
        expected_dates = set(range(0, number_days.days))
        actual_dates = {point.days for point in self.processed_data}
        missing_dates = expected_dates - actual_dates
        logger.debug(f"Expected dates: {expected_dates}")
        logger.debug(f"Actual dates: {actual_dates}")
        logger.debug(f"Found missing days: {missing_dates}")
        for missing in missing_dates:
            self.raw_data.append(RawDataPoint(
                date=self.start_date + timedelta(days=missing),
                value=0.0,
            ))

    def _sort_data(self, offset_days):
        unsorted = []
        for point in self.raw_data:
            day = point.date - self.start_date + timedelta(days=offset_days)
            unsorted.append(ProcessedDataPoint(day.days, point.value))
        self.processed_data = sorted(unsorted, key=lambda x: x.days)


    def plot_data(self, ylabel: str, graph_file: Path | None = None):
        if not self.processed_data:
            raise DataPointError("Data has not been processed yet")
        dates, values = self.convert_processed_to_list()
        pl.plot(dates, values, 'k.')
        pl.xlabel(f"Days from {self.start_date} - 1")
        pl.ylabel(ylabel)
        if graph_file:
            pl.savefig(graph_file)
        else:
            pl.show()

    def convert_processed_to_list(self) -> tuple[list[int], list[float]]:
        days = [point.days for point in self.processed_data]
        values = [point.value for point in self.processed_data]
        return days, values
    
    def retrieve_dates(self) -> list[date]:
        return [point.date for point in self.raw_data]