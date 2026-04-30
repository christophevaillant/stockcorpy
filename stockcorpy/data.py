from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from pickle import dump, load
from pathlib import Path
import logging

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
    @abstractmethod
    def __init__(self):
        self.raw_data: list[RawDataPoint] = []
        self.processed_data: list[ProcessedDataPoint] | None = None
        self.start_date: date = date.today()
        self.end_date: date = date.today()

    @classmethod
    def load_from_file(cls, filepath: Path):
        obj = cls()
        with open(filepath, "r") as datafile:
            obj.raw_data = load(datafile)
        existing_dates = obj.retrieve_dates()
        obj.start_date = min(existing_dates)
        obj.end_date = max(existing_dates)
        return obj

    def save_to_file(self, filepath: Path):
        with open(filepath, "w") as datafile:
            dump(self.raw_data, datafile)

    @abstractmethod
    def create_data(self, number_of_days: int):
        pass

    @abstractmethod
    def process_data(self, offset_days: int = 0):
        dates = [point.date for point in self.raw_data]
        start_date = min(dates)
        end_date = max(dates)
        length = end_date - start_date
        logger.info(f"Found data points between {start_date} and {end_date}, a total of {length.days} days")
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