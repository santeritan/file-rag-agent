# framework/components/ingestion.py

from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
)
import os

class FileIngestionPipeline:
    """
    Loads and chunks supported file types.
    Core types: PDF, TXT, CSV
    Optional types: DOCX, XLSX, HTML, MD
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Core loaders - always available
        self.LOADER_MAP = {
            "pdf":  PyPDFLoader,
            "text": TextLoader,
            "csv":  CSVLoader,
        }

        # Optional loaders
        self._load_optional_loaders()

    def _load_optional_loaders(self):
        silent = os.environ.get("SILENT_MODE", "false") == "true"

        try:
            from langchain_community.document_loaders import Docx2txtLoader
            self.LOADER_MAP["docx"] = Docx2txtLoader
            if not silent:
                print("✅ DOCX support enabled")
        except ImportError:
            if not silent:
                print("⚠️  DOCX not available")

        try:
            from langchain_community.document_loaders import (
                UnstructuredExcelLoader
            )
            self.LOADER_MAP["excel"] = UnstructuredExcelLoader
            if not silent:
                print("✅ Excel support enabled")
        except ImportError:
            if not silent:
                print("⚠️  Excel not available")

        try:
            from langchain_community.document_loaders import (
                UnstructuredHTMLLoader
            )
            self.LOADER_MAP["html"] = UnstructuredHTMLLoader
            if not silent:
                print("✅ HTML support enabled")
        except ImportError:
            if not silent:
                print("⚠️  HTML not available")

        try:
            from langchain_community.document_loaders import (
                UnstructuredMarkdownLoader
            )
            self.LOADER_MAP["markdown"] = UnstructuredMarkdownLoader
            if not silent:
                print("✅ Markdown support enabled")
        except ImportError:
            self.LOADER_MAP["markdown"] = TextLoader
            if not silent:
                print("✅ Markdown support enabled (text fallback)")

    def get_supported_types(self) -> List[str]:
        return list(self.LOADER_MAP.keys())

    def load_file(
        self,
        file_path: str,
        file_type: str,
        file_id: str,
        original_name: str
    ) -> List[Document]:
        """Load a single file into Documents"""

        loader_class = self.LOADER_MAP.get(file_type)
        if not loader_class:
            raise ValueError(
                f"No loader for type: {file_type}\n"
                f"Supported types: {self.get_supported_types()}"
            )

        loader = loader_class(file_path)
        documents = loader.load()

        # Add metadata to each document
        for doc in documents:
            doc.metadata.update({
                "file_id":   file_id,
                "file_name": original_name,
                "file_type": file_type,
            })

        return documents

    # framework/components/ingestion.py
    # Find process_file and process_records

    def process_file(self, file_path, file_type,
                    file_id, original_name):
        silent = os.environ.get("SILENT_MODE") == "true"

        documents = self.load_file(
            file_path, file_type, file_id, original_name
        )
        chunks = self.splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        if not silent:
            print(f"   📄 {original_name}: "
                f"{len(documents)} pages → {len(chunks)} chunks")

        return chunks

    def process_records(self, file_records: list) -> List[Document]:
        silent = os.environ.get("SILENT_MODE") == "true"
        all_chunks = []

        if not silent:
            print("\n🔄 Processing files...")

        for record in file_records:
            try:
                chunks = self.process_file(
                    file_path=record.file_path,
                    file_type=record.file_type,
                    file_id=record.file_id,
                    original_name=record.original_name
                )
                all_chunks.extend(chunks)
            except Exception as e:
                if not silent:
                    print(f"   ❌ Failed: {record.original_name} → {e}")

        if not silent:
            print(f"\n✅ Total chunks ready: {len(all_chunks)}\n")

        return all_chunks