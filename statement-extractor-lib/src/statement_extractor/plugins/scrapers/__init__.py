"""
Scraper plugins for fetching content from URLs.

Built-in scrapers:
- http_scraper: Default HTTP scraper using httpx with retries
"""

from .http import HttpScraperPlugin

__all__ = ["HttpScraperPlugin"]
