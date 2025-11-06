"""Cristin API ingestion utilities for staff research results."""

from __future__ import annotations

__all__ = [
    "CristinClient",
    "ResultRecord",
    "sync_cristin_results",
]

from .client import CristinClient
from .models import ResultRecord
from .sync import sync_cristin_results
