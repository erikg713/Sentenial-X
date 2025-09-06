# -*- coding: utf-8 -*-
"""
Base class for semantic models in Sentenial-X
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseSemanticModel(ABC):
    """
    Abstract base class for semantic models.
    """

    @abstractmethod
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze text and return structured results.
        Must be implemented by all subclasses.
        """
        raise NotImplementedError
