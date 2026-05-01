from pathlib import Path
import logging

from numpy import corrcoef

from ..core.stock import Stock
from ..core.news import Keyword
from ..core.utils import load_sentiment_classifier
from .model import Model

logger = logging.getLogger("stock-news")

class StockNews(Model):
    initial_number_days = 30
    threshold_correlation = 0.7

    def __init__(self, model_name, stock_filename: Path, keyword_filename: Path):
        super().__init__(model_name)
        self.stocknames = stock_filename.read_text().splitlines()
        self.keywordnames = keyword_filename.read_text().splitlines()
        self.stocks = []
        self.keywords = []
        self.stocks_to_keywords = {}
        self.sentiment_classifier = load_sentiment_classifier()

    def create_model(self):
        logger.info("--------------------------------------")
        logger.info("Loading stocks:")
        self.stocks = self._load_data_from_list(Stock, self.stocknames)

        logger.info("--------------------------------------")
        logger.info("Loading keywords:")
        self.keywords = self._load_data_from_list(Keyword, self.keywordnames, classifier = self.sentiment_classifier)

    def update_model(self):
        for data in self.stocks + self.keywords:
            data.create_data(self.initial_number_days)
            data.process_data()

    def _load_data_from_list(self, cls, data_names, **kwargs):
        data_list = []
        for name in data_names:
            logger.info(f"Loading {name}")
            new_data = cls(name, **kwargs)
            new_data.create_data(self.initial_number_days)
            new_data.process_data()
            data_list.append(new_data)
        return data_list

    def train_model(self):
        for stock in self.stocks():
            correlated_keywords = self._retrieve_correlated_keywords(stock)
            if len(correlated_keywords) > 0:
                self.stocks_to_keywords[stock.name] = correlated_keywords
    
    def _retrieve_correlated_keywords(self, stock: Stock) -> list[str]:
        logger.info(f"Finding keywords for stock {stock.name}")
        correlated_keywords = []
        for keyword in self.keywords:
            correlation = corrcoef(stock.processed_data, keyword.processed_data)
            if correlation > self.threshold_correlation:
                correlated_keywords.append(keyword.name)
                logger.info(f"Found keyword {keyword.name} with correlation {correlation}")
        return correlated_keywords
    
    def predict_next_day(self):
        raise NotImplementedError("This is just a placeholder")