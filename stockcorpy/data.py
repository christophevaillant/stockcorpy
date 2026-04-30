from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
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
        self.raw_data: list[RawDataPoint] | None = None
        self.processed_data: list[ProcessedDataPoint] | None = None
        self.start_date: date = date.today()
        self.end_date: date = date.today()

    def load_from_file(self, filepath: Path):
        with open(filepath, "r") as datafile:
            self.raw_data = load(datafile)

    def save_to_file(self, filepath: Path):
        with open(filepath, "w") as datafile:
            dump(self.raw_data, datafile)

    @abstractmethod
    def create_data(self):
        pass

    @abstractmethod
    def process_data(self, offset: int = 0):
        dates = [point.date for point in self.raw_data]
        start_date = min(dates)
        end_date = max(dates)
        length = end_date - start_date
        logger.info(f"Found data points between {start_date} and {end_date}, a total of {length.days} days")
        unsorted = []
        for point in self.raw_data:
            day = point.date - self.start_date
            unsorted.append(ProcessedDataPoint(day, point.value))
        self.processed_data = sorted(unsorted, key=lambda x: x.day)


    @abstractmethod
    def plot_data(self):
        pass

    def convert_processed_to_list(self) -> tuple[list[int], list[float]]:
        days = [point.day for point in self.processed_data]
        values = [point.value for point in self.processed_data]
        return days, values