import numpy as np
import pylab as pl
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import os

from newsapi import NewsApiClient
from newspaper import Article
from newspaper.exceptions import ArticleException, ArticleBinaryDataException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .data import Data, RawDataPoint
from .utils import round_to_nearest_day

logger = logging.getLogger("news")

API_DAYS_LIMIT = 30

class Keyword(Data):
    def __init__(self, keyword: str, storage_file: Path | None = None):
        super().__init__()
        self.keyword = keyword
        if storage_file:
            self.load_from_file(storage_file)
        else:
            self.create_data(API_DAYS_LIMIT)

    def create_data(self, number_of_days):
        if number_of_days > API_DAYS_LIMIT:
            logger.warning("Number of days exceeds allowed value from provider, defaulting to 30 days")
            number_of_days = API_DAYS_LIMIT
        stored_days = (date.today() - self.end_date).days
        if stored_days > 0:
            number_of_days = stored_days

        self._load_analyzers()
        articles = self._get_articles(date.today() - timedelta(days=number_of_days))

        logger.info(f"found {len(articles)} new articles")
        article_scores = {}
        for article in articles:
            article_date = round_to_nearest_day(
                datetime.fromisoformat(article['publishedAt'])
            )
            existing_dates = self.retrieve_dates()
            if article_date not in existing_dates:
                average = self._analyze_article(article)
                if article_date in article_scores:
                    article_scores[article_date] += average
                else:
                    article_scores[article_date] = average
        for article_date, score in article_scores.items():
            self.raw_data.append(RawDataPoint(
                date=article_date,
                value=score,
            ))

    def process_data(self, offset_days = 0):
        return super().process_data(offset_days)

    def _load_analyzers(self):
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,        # Safe margin under the 512 limit
            chunk_overlap=80,      # 20% overlap to maintain context
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.sentiment_mapping = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
        }

    def _get_articles(self, date_range: date) -> list:
        newsapi = NewsApiClient(api_key=os.environ["NEWS_API_KEY"])
        keyword_responses = newsapi.get_everything(q='AI integration', from_param=date_range.isoformat(), language='en')
        return keyword_responses['articles']

    def _analyze_article(self, article: list) -> float:
        try:
            content = Article(article['url'])
            content.download()
            content.parse()
            chunks = self.text_splitter.split_text(content.text)
            chunk_scores = []
            for i, chunk in enumerate(chunks):
                analysis = self.classifier(chunk)
                logger.debug(f'chunk {i} gives analysis {analysis[0]}')
                chunk_scores.append(self.sentiment_mapping[analysis[0]['label']])
            return np.average(chunk_scores)
        except (ArticleException, ArticleBinaryDataException) :
            logger.debug(f'article {article["url"]} could not be downloaded')
            return 0.0

    def plot_data(self, graph_file: Path | None = None):
        super().plot_data("Keyword score", graph_file=graph_file)
