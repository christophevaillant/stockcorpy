from pathlib import Path
import logging
from time import sleep
from copy import deepcopy

from scipy.stats import pearsonr

from ..core.stock import Stock, StockProcessingMode
from ..core.news import Keyword
from ..core.data import DataPointError
from .model import Model

logger = logging.getLogger("stock-news")

class StockNews(Model):
    initial_number_days = 50
    threshold_correlation = 0.5

    def __init__(self, model_name, stock_filename: Path, keyword_filename: Path):
        super().__init__(model_name)
        self.stocknames = stock_filename.read_text().splitlines()
        self.keywordnames = keyword_filename.read_text().splitlines()
        self.stocks = []
        self.keywords = []
        self.stocks_to_keywords = {}

    def create_model(self):
        logger.info("--------------------------------------")
        logger.info("Loading stocks:")
        self.stocks = self._load_data_from_list(Stock, self.stocknames)

        logger.info("--------------------------------------")
        logger.info("Loading keywords:")
        self.keywords = self._load_data_from_list(Keyword, self.keywordnames)

    def update_model(self):
        for data in self.stocks + self.keywords:
            data.create_data(self.initial_number_days)
            data.process_data()

    def _load_data_from_list(self, cls, data_names, **kwargs):
        data_list = []
        for name in data_names:
            logger.info(f"Loading {name}")
            try:
                new_data = cls(name, **kwargs)
                new_data.create_data(self.initial_number_days)
                new_data.process_data()
                data_list.append(deepcopy(new_data))
            except DataPointError:
                logger.info(f'Issue with data point {name}, skipping...')
            sleep(15)
        return data_list
    
    def process_model_data(self):
        all_dates = [point.date for data in self.stocks + self.keywords for point in data.raw_data]
        self.start_date = min(all_dates)
        self.end_date = max(all_dates)
        logger.debug(f"Found dates between {self.start_date} and {self.end_date}")
        for data in self.keywords:
            data.start_date = self.start_date
            data.end_date = self.end_date
            data.process_data()
        for data in self.stocks:
            data.start_date = self.start_date
            data.end_date = self.end_date
            data.process_data(StockProcessingMode.DIFFERENCE)

    def train_model(self):
        for stock in self.stocks:
            correlated_keywords = self._retrieve_correlated_keywords(stock)
            if len(correlated_keywords) > 0:
                self.stocks_to_keywords[stock.name] = correlated_keywords
    
    def _retrieve_correlated_keywords(self, stock: Stock) -> list[str]:
        logger.info(f"Finding keywords for stock {stock.name}")
        correlated_keywords = []
        stock_days = {point.days for point in stock.processed_data}
        for keyword in self.keywords:
            keyword_days = {point.days for point in keyword.processed_data}
            joint_days = stock_days & keyword_days
            if len(joint_days) > 5:
                logger.debug(f"Found {len(joint_days)} joint days")
                stock_data = [point.value for point in stock.processed_data if point.days in joint_days]
                keyword_data = [point.value for point in keyword.processed_data if point.days in joint_days]
                if len(set(stock_data)) == 1 or len(set(keyword_data)) == 1:
                    logger.debug(f"Skipping {keyword.keyword}: zero variance")
                    continue
                correlation = pearsonr(stock_data, keyword_data)
                logger.debug(f"Stock {stock.name} keyword {keyword.keyword}: {correlation.statistic}, with {correlation.pvalue}")
                if abs(correlation.statistic) >= self.threshold_correlation:
                    correlated_keywords.append(keyword.keyword)
                    logger.info(f"Found keyword {keyword.keyword} with correlation {correlation.statistic}")
        return correlated_keywords
    
    def predict_next_day(self):
        raise NotImplementedError("This is just a placeholder")