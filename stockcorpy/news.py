import numpy as np
import pylab as pl
import logging
from datetime import datetime, date
from pathlib import Path
import os

from newsapi import NewsApiClient
from newspaper import Article
from newspaper.exceptions import ArticleException, ArticleBinaryDataException
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .data import Data, DataPointError
from .utils import round_to_nearest_day

logger = logging.getLogger("news")

class Keyword(Data):
    def __init__(self, keyword: str, storage_file: Path | None = None):
        self.keyword = keyword
        if storage_file:
            self.load_from_file(storage_file)
        else:
            self.create_keyword_data()

    def create_data(self, number_of_days):
        if number_of_days >= 30:
            logger.warning("Number of days exceeds allowed value from provider, defaulting to 30 days")
            number_of_days = 30
        stored_days = date.today() - self.end_date
        if stored_days > 0:
            number_of_days = stored_days

        self._load_analyzers()
        articles = self._get_articles(date.today - number_of_days)

        logger.info(f"found {len(articles)} new articles")
        for article in articles:
            date = round_to_nearest_day(
                datetime.fromisoformat(article['publishedAt'])
            )
            existing_dates = self.retrieve_dates()
            if date not in existing_dates:
                average = self._analyze_article(article)
            self.raw_data[date] = average

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
        newsapi = NewsApiClient(api_key=os.environ("NEWS_API_KEY"))
        keyword_responses = newsapi.get_everything(q='AI integration', from=date_range.isoformat(), language='en')
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
                print(f'chunk {i} gives analysis {analysis[0]}')
                chunk_scores.append(self.sentiment_mapping[analysis[0]['label']])
            return np.average(chunk_scores)
        except (ArticleException, ArticleBinaryDataException) :
            logger.debug(f'article {article["url"]} could not be downloaded')
            return 0.0

    def plot_data(self, graph_file: Path | None = None):
        if not self.processed_data:
            raise DataPointError("Data has not been processed yet")
        dates, values = self.convert_processed_to_list()
        pl.plot(dates, values, 'k.')
        pl.xlabel(f"Days from {self.start_date}")
        pl.ylabel("Keyword score")
        if graph_file:
            pl.savefig(graph_file)
        else:
            pl.show()
