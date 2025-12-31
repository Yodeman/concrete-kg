"""PDF Processing module for extracting text from research papers."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Union

import fitz  # PyMuPDF


@dataclass
class Section:
    """Represents a section of a document."""

    title: str
    content: str
    page_start: int
    page_end: int


@dataclass
class TableData:
    """Represents extracted table data."""

    headers: List[str]
    rows: List[List[str]]
    page: int
    caption: Optional[str] = None


class PDFProcessor:
    """Processes PDF documents to extract text and structured content."""

    def __init__(self):
        """Initialize the PDF processor."""
        pass

    def extract_text(self, pdf_source: Union[str, Path, BinaryIO, bytes]) -> str:
        """
        Extract all text from a PDF document.

        Args:
            pdf_source: Path to PDF file, file-like object, or bytes

        Returns:
            Extracted text as a string
        """
        doc = self._open_document(pdf_source)
        try:
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_parts.append(page.get_text())
            return "\n\n".join(text_parts)
        finally:
            doc.close()

    def extract_text_by_page(
        self, pdf_source: Union[str, Path, BinaryIO, bytes]
    ) -> List[str]:
        """
        Extract text from each page separately.

        Args:
            pdf_source: Path to PDF file, file-like object, or bytes

        Returns:
            List of text content for each page
        """
        doc = self._open_document(pdf_source)
        try:
            pages = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pages.append(page.get_text())
            return pages
        finally:
            doc.close()

    def extract_sections(
        self, pdf_source: Union[str, Path, BinaryIO, bytes]
    ) -> List[Section]:
        """
        Extract sections based on text formatting (headings).

        Args:
            pdf_source: Path to PDF file, file-like object, or bytes

        Returns:
            List of Section objects
        """
        doc = self._open_document(pdf_source)
        try:
            sections = []
            current_section = None
            current_content = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                for block in blocks:
                    if "lines" not in block:
                        continue

                    for line in block["lines"]:
                        text = "".join(span["text"] for span in line["spans"])

                        # Detect potential section headers (larger font, bold)
                        is_header = False
                        if line["spans"]:
                            avg_size = sum(
                                span["size"] for span in line["spans"]
                            ) / len(line["spans"])
                            is_bold = any(
                                "bold" in span.get("font", "").lower()
                                for span in line["spans"]
                            )
                            is_header = avg_size > 12 or is_bold

                        if (
                            is_header
                            and len(text.strip()) > 2
                            and len(text.strip()) < 100
                        ):
                            # Save previous section
                            if current_section:
                                current_section.content = "\n".join(current_content)
                                current_section.page_end = page_num
                                sections.append(current_section)

                            # Start new section
                            current_section = Section(
                                title=text.strip(),
                                content="",
                                page_start=page_num,
                                page_end=page_num,
                            )
                            current_content = []
                        elif current_section:
                            current_content.append(text)

            # Save last section
            if current_section:
                current_section.content = "\n".join(current_content)
                current_section.page_end = len(doc) - 1
                sections.append(current_section)

            return sections
        finally:
            doc.close()

    def extract_tables(
        self, pdf_source: Union[str, Path, BinaryIO, bytes]
    ) -> List[TableData]:
        """
        Extract tables from PDF (basic implementation using text blocks).

        Args:
            pdf_source: Path to PDF file, file-like object, or bytes

        Returns:
            List of TableData objects
        """
        doc = self._open_document(pdf_source)
        try:
            tables = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                # Use built-in table finder if available (PyMuPDF 1.23+)
                try:
                    page_tables = page.find_tables()
                    for table in page_tables:
                        extracted = table.extract()
                        if extracted and len(extracted) > 1:
                            tables.append(
                                TableData(
                                    headers=extracted[0] if extracted else [],
                                    rows=extracted[1:] if len(extracted) > 1 else [],
                                    page=page_num,
                                )
                            )
                except AttributeError:
                    # Fallback for older PyMuPDF versions
                    pass

            return tables
        finally:
            doc.close()

    def get_metadata(self, pdf_source: Union[str, Path, BinaryIO, bytes]) -> dict:
        """
        Extract PDF metadata.

        Args:
            pdf_source: Path to PDF file, file-like object, or bytes

        Returns:
            Dictionary of metadata
        """
        doc = self._open_document(pdf_source)
        try:
            return {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "page_count": len(doc),
                "creator": doc.metadata.get("creator", ""),
            }
        finally:
            doc.close()

    def _open_document(
        self, pdf_source: Union[str, Path, BinaryIO, bytes]
    ) -> fitz.Document:
        """
        Open a PDF document from various source types.

        Args:
            pdf_source: Path, file-like object, or bytes

        Returns:
            PyMuPDF Document object
        """
        if isinstance(pdf_source, (str, Path)):
            return fitz.open(str(pdf_source))
        elif isinstance(pdf_source, bytes):
            return fitz.open(stream=pdf_source, filetype="pdf")
        else:
            # File-like object (e.g., from Streamlit uploader)
            content = pdf_source.read()
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            return fitz.open(stream=content, filetype="pdf")
