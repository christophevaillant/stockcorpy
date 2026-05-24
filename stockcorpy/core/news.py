import numpy as np
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from requests.exceptions import ConnectTimeout, RequestException
from newspaper import Article
from newspaper.exceptions import ArticleException, ArticleBinaryDataException

from .data import Data, RawDataPoint
from .utils import round_to_nearest_day, load_sentiment_classifier

logger = logging.getLogger("news")

API_DAYS_LIMIT = 50

BREAKING_NEWS_DOMAINS = [
    'apnews.com',
    'cnbc.com',
    'axios.com',
]

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
                datetime.strptime(article['seendate'], "%Y%m%dT%H%M%SZ")
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
                value=score / article_numbers[article_date],
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
        domain_filter = ' OR '.join(f'domain:{d}' for d in BREAKING_NEWS_DOMAINS)
        query = f'"{self.keyword}" ({domain_filter})'
        logger.debug(query)
        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": 250,
            "startdatetime": date_range.strftime("%Y%m%d%H%M%S"),
            "enddatetime": date.today().strftime("%Y%m%d%H%M%S"),
        }
        for attempt in range(7):
            try:
                response = requests.get(
                    "https://api.gdeltproject.org/api/v2/doc/doc",
                    params=params,
                    timeout=30,  # seconds
                )
                if response.status_code == 200:
                    if not response.content:
                        logger.warning("GDELT returned empty response body, query may be malformed or too long")
                        return []
                    logger.debug(f"GDELT raw response: {response.content[:500]}")
                    return response.json().get("articles", [])
                elif response.status_code == 429:
                    wait = 2 ** attempt * 15
                    logger.warning(f"GDELT rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.warning(f"Unexpected response {response.status_code} from GDELT, skipping")
                    return []
            except ConnectTimeout:
                wait = 2 ** attempt * 15
                logger.warning(f"GDELT connection timed out, retrying in {wait}s...")
                time.sleep(wait)
            except RequestException as e:
                logger.error(f"GDELT request failed: {e}")
                return []

        logger.error("GDELT retries exhausted")
        return []

    def _analyze_article(self, article: list, classifier, text_splitter, sentiment_mapping) -> float:
        try:
            content = Article(article['url'], fetch_images=False)
            content.download()
            content.parse()
            if not content.text or not content.text.strip():
                logger.debug(f'article {article["url"]} has no extractable text')
                return 0.0
            chunks = text_splitter.split_text(content.text)
            if not chunks:
                logger.debug(f'article {article["url"]} produced no chunks')
                return 0.0
            chunk_scores = []
            for i, chunk in enumerate(chunks):
                analysis = classifier(chunk)
                logger.debug(f'chunk {i} gives analysis {analysis[0]}')
                chunk_scores.append(sentiment_mapping[analysis[0]['label']] * analysis[0]['score'])
            return np.average(chunk_scores)
        except (ArticleException, ArticleBinaryDataException, ModuleNotFoundError):
            logger.debug(f'article {article["url"]} could not be downloaded')
            return 0.0

    def plot_data(self, graph_file: Path | None = None):
        super().plot_data("Keyword score", graph_file=graph_file)
