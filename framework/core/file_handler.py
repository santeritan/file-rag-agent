# framework/core/file_handler.py

import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class FileRecord:
    """Tracks each uploaded file"""
    file_id: str
    original_name: str
    file_path: str
    file_type: str
    file_size: int
    upload_time: str
    chunk_count: int = 0
    is_indexed: bool = False
    metadata: Dict = field(default_factory=dict)


class FileHandler:
    """
    Manages file uploads, validation, and tracking.
    Supports: PDF, TXT, DOCX, CSV, JSON, MD, XLSX, HTML
    """

    SUPPORTED_TYPES = {
        ".pdf":  "pdf",
        ".txt":  "text",
        ".docx": "docx",
        ".csv":  "csv",
        ".json": "json",
        ".md":   "markdown",
        ".xlsx": "excel",
        ".html": "html",
        ".htm":  "html",
        ".pptx": "pptx",
    }

    MAX_FILE_SIZE_MB = 50

    def __init__(self, upload_dir: str = "uploaded_files"):
        # Fix 1: Keep upload_dir as Path object internally
        # but store as str for the dataclass
        self.upload_dir: Path = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.file_registry: Dict[str, FileRecord] = {}

    # ─── Upload & Validate ────────────────────────────────────

    # framework/core/file_handler.py
    # Find the add_file method and replace it

    def add_file(self, file_path: str) -> FileRecord:
        """
        Add a file. Does NOT copy it - uses the original location.
        This prevents accumulating duplicate files.
        """
        path = Path(file_path)

        # Validate
        self._validate_file(path)

        # Generate unique ID from content hash
        file_id = self._generate_file_id(path)

        # Check for duplicates
        if file_id in self.file_registry:
            print(f"⚠️  File already loaded: {path.name}")
            return self.file_registry[file_id]

        # ── KEY CHANGE: Don't copy the file ──────────────────
        # Just use the original path directly
        # This prevents duplicates from accumulating
        
        # Create record with original path
        record = FileRecord(
            file_id=file_id,
            original_name=path.name,
            file_path=str(path.absolute()),  # Use absolute path to original
            file_type=self.SUPPORTED_TYPES[path.suffix.lower()],
            file_size=path.stat().st_size,
            upload_time=datetime.now().isoformat(),
            metadata={"original_path": str(path)}
        )

        self.file_registry[file_id] = record
        print(f"✅ Added file: {path.name} (ID: {file_id[:8]}...)")

        return record

    def add_files(self, file_paths: List[str]) -> List[FileRecord]:
        """Add multiple files at once"""
        records = []
        for path in file_paths:
            try:
                record = self.add_file(path)
                records.append(record)
            except Exception as e:
                print(f"❌ Failed to add {path}: {e}")

        self._print_summary(records)
        return records

    def add_directory(
        self,
        dir_path: str,
        recursive: bool = False
    ) -> List[FileRecord]:
        """Add all supported files from a directory"""

        # Fix 4: Convert to Path object for directory operations
        dir_as_path: Path = Path(dir_path)

        # Fix 5: Use Path object for is_dir() check
        if not dir_as_path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        # Fix 6: Use Path object for glob
        pattern = "**/*" if recursive else "*"
        file_paths: List[str] = [
            str(f) for f in dir_as_path.glob(pattern)
            if f.is_file()
            and f.suffix.lower() in self.SUPPORTED_TYPES
        ]

        print(f"📁 Found {len(file_paths)} supported files in {dir_path}")
        return self.add_files(file_paths)

    def remove_file(self, file_id: str) -> bool:
        """Remove a file from the agent knowledge"""
        if file_id not in self.file_registry:
            print(f"❌ File ID not found: {file_id}")
            return False

        record = self.file_registry[file_id]

        # Fix 7: Convert str path back to Path for file operations
        path: Path = Path(record.file_path)
        if path.exists():
            path.unlink()

        del self.file_registry[file_id]
        print(f"🗑️  Removed: {record.original_name}")
        return True

    def clear_all_files(self) -> None:
        """Remove all files from agent knowledge"""
        for file_id in list(self.file_registry.keys()):
            self.remove_file(file_id)
        print("🧹 All files cleared")

    # ─── Querying Registry ────────────────────────────────────

    def get_file(self, file_id: str) -> Optional[FileRecord]:
        return self.file_registry.get(file_id)

    def list_files(self) -> List[FileRecord]:
        return list(self.file_registry.values())

    def get_indexed_files(self) -> List[FileRecord]:
        return [
            f for f in self.file_registry.values()
            if f.is_indexed
        ]

    def get_unindexed_files(self) -> List[FileRecord]:
        return [
            f for f in self.file_registry.values()
            if not f.is_indexed
        ]

    def mark_indexed(self, file_id: str, chunk_count: int) -> None:
        if file_id in self.file_registry:
            self.file_registry[file_id].is_indexed = True
            self.file_registry[file_id].chunk_count = chunk_count

    def print_status(self) -> None:
        """Print current file status table"""
        files = self.list_files()
        if not files:
            print("📭 No files loaded")
            return

        print("\n" + "="*70)
        print(
            f"{'FILE NAME':<30} {'TYPE':<10} "
            f"{'SIZE':<10} {'STATUS':<10}"
        )
        print("="*70)

        for f in files:
            size_str = self._format_size(f.file_size)
            status = "✅ Indexed" if f.is_indexed else "⏳ Pending"
            print(
                f"{f.original_name:<30} {f.file_type:<10} "
                f"{size_str:<10} {status}"
            )

        print("="*70)
        print(
            f"Total: {len(files)} files | "
            f"Indexed: {len(self.get_indexed_files())} | "
            f"Pending: {len(self.get_unindexed_files())}"
        )
        print("="*70 + "\n")

    # ─── Private Helpers ──────────────────────────────────────

    def _validate_file(self, path: Path) -> None:
        """Validate file exists, is supported, and not too large"""

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {path}")

        if path.suffix.lower() not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unsupported type: {path.suffix}\n"
                f"Supported: {list(self.SUPPORTED_TYPES.keys())}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB "
                f"(max: {self.MAX_FILE_SIZE_MB}MB)"
            )

    def _generate_file_id(self, path: Path) -> str:
        """Generate ID based on file content hash"""
        hasher = hashlib.md5()
        with open(str(path), "rb") as f:
            hasher.update(f.read())
        return hasher.hexdigest()

    def _format_size(self, size_bytes: int) -> str:
        """Format file size to human readable string"""
        if size_bytes < 1024:
            return f"{size_bytes}B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f}KB"
        else:
            return f"{size_bytes/1024**2:.1f}MB"

    def _print_summary(self, records: List[FileRecord]) -> None:
        success = len(records)
        print(f"\n📊 Upload Summary: {success} files ready for indexing")