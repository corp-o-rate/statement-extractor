"""
PDF parser plugins for extracting text from PDF files.

Built-in parsers:
- pypdf_parser: Default PDF parser using PyMuPDF with optional OCR
"""

from .pypdf import PyPDFParserPlugin

__all__ = ["PyPDFParserPlugin"]
