# core/ingester/__init__.py
from .data_processor import process_data
from .database_connector import connect

# Elsewhere in the project
from core.ingester import process_data, connect

