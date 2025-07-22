import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import aiosqlite
import json
from config import DB_PATH
from logger import setup_logger

logger = setup_logger('ml_classifier')
