"""
libs/core/plugins/pdf_meta.py

PDF metadata extraction utility for Sentenial-X platform.
Supports extracting basic info and optional custom metadata fields.
"""

from typing import Dict, Optional
from pathlib import Path
from PyPDF2 import PdfReader
import logging

logger = logging.getLogger("sentenialx.pdf_meta")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def extract_pdf_metadata(file_path: str) -> Optional[Dict[str, str]]:
    """
    Extract metadata from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        dict: Dictionary containing PDF metadata.
        None: If the PDF cannot be read or file is invalid.
    """
    pdf_path = Path(file_path)
    if not pdf_path.is_file():
        logger.error("PDF file not found: %s", file_path)
        return None

    try:
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata
        metadata_dict = {key[1:]: str(value) for key, value in meta.items() if value is not None}
        logger.info("Extracted metadata from %s", file_path)
        return metadata_dict
    except Exception as e:
        logger.exception("Failed to extract PDF metadata: %s", e)
        return None


if __name__ == "__main__":
    # Example usage
    sample_pdf = "example.pdf"
    metadata = extract_pdf_metadata(sample_pdf)
    if metadata:
        print("PDF Metadata:")
        for k, v in metadata.items():
            print(f"{k}: {v}")
    else:
        print("Failed to extract metadata.")
