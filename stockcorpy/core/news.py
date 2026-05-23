import numpy as np
import pylab as pl
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import os

from newsapi import NewsApiClient
from newspaper import Article
from newspaper.exceptions import ArticleException, ArticleBinaryDataException
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .data import Data, RawDataPoint
from .utils import round_to_nearest_day, load_sentiment_classifier

logger = logging.getLogger("news")

API_DAYS_LIMIT = 30

class Keyword(Data):
    def __init__(self, keyword: str):
        super().__init__()
        self.keyword = keyword

    def create_data(self, number_of_days):
        if number_of_days > API_DAYS_LIMIT:
            logger.warning("Number of days exceeds allowed value from provider, defaulting to 30 days")
            number_of_days = API_DAYS_LIMIT
        stored_days = (date.today() - self.end_date).days
        if stored_days > 0:
            number_of_days = stored_days

        classifier, text_splitter, sentiment_mapping = self._load_analyzers()
        articles = self._get_articles(date.today() - timedelta(days=number_of_days))

        logger.info(f"found {len(articles)} new articles")
        article_scores = {}
        article_numbers = {}
        for article in articles:
            article_date = round_to_nearest_day(
                datetime.fromisoformat(article['publishedAt'])
            )
            existing_dates = self.retrieve_dates()
            if article_date not in existing_dates:
                average = self._analyze_article(article, classifier, text_splitter, sentiment_mapping)
                if article_date in article_scores:
                    article_scores[article_date] += average
                    article_numbers[article_date] += 1
                else:
                    article_scores[article_date] = average
                    article_numbers[article_date] = 1
        for article_date, score in article_scores.items():
            self.raw_data.append(RawDataPoint(
                date=article_date,
                value=score * article_numbers[article_date],
            ))
        super().create_data(number_of_days)

    def process_data(self, offset_days = 0):
        return super().process_data(offset_days)

    def _load_analyzers(self):
        classifier = load_sentiment_classifier()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,        # Safe margin under the 512 limit
            chunk_overlap=80,      # 20% overlap to maintain context
            separators=["\n\n", "\n", ".", " ", ""]
            )
        sentiment_mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.1,
        }
        return classifier, text_splitter, sentiment_mapping

    def _get_articles(self, date_range: date) -> list:
        newsapi = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
        keyword_responses = newsapi.get_everything(q=self.keyword, from_param=date_range.isoformat(), language='en')
        return keyword_responses['articles']

    def _analyze_article(self, article: list, classifier, text_splitter, sentiment_mapping) -> float:
        try:
            content = Article(article['url'], fetch_images=False)
            content.download()
            content.parse()
            chunks = text_splitter.split_text(content.text)
            chunk_scores = []
            for i, chunk in enumerate(chunks):
                analysis = classifier(chunk)
                logger.debug(f'chunk {i} gives analysis {analysis[0]}')
                chunk_scores.append(sentiment_mapping[analysis[0]['label']])
            return np.average(chunk_scores)
        except (ArticleException, ArticleBinaryDataException, ModuleNotFoundError) :
            logger.debug(f'article {article["url"]} could not be downloaded')
            return 0.0

    def plot_data(self, graph_file: Path | None = None):
        super().plot_data("Keyword score", graph_file=graph_file)
