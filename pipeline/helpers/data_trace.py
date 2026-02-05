"""
DataTraceHelper - temporary helper for DataFrame tracing in research/debugging mode.

Singleton pattern for global access from any system class.
CSV-only storage for readability, auto-incrementing runs, fail-fast error handling.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

__version__ = "1.0.0"


class TraceError(Exception):
    """Custom exception for DataTraceHelper errors."""
    pass


class DataTraceHelper:
    """
    Singleton helper for DataFrame tracing in research/debugging pipeline.

    Thread-safety: Not required (sequential execution)
    Storage: CSV for DataFrames, JSON for context dicts
    Error handling: Fail-fast with TraceError exceptions

    Usage:
        # Initialization (once in main/factory)
        trace = DataTraceHelper(base_dir="data/traces")
        trace.start_run()

        # Manual save
        trace.save_df(df, "checkpoint_name")
        trace.save_context(context_dict, "pipeline_context")

        # Finalize
        trace.finalize()
    """

    def __init__(self, base_dir: str = "data/traces"):
        """
        Initialize helper.

        Args:
            base_dir: Root directory for traces (created if not exists)
        """
        self._base_dir = Path(base_dir)
        self._current_run_dir = None  # Path to run_XXX/
        self._current_run_id = None  # String like "001"
        self._file_counter = 0  # Auto-increment counter
        self._metadata = {  # In-memory metadata
            "run_id": None,
            "timestamp_start": None,
            "timestamp_end": None,
            "total_files": 0,
            "files": []
        }
        self._last_file_path = None  # Cache for chaining

        # Create base directory if not exists
        self._base_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(f"DataTraceHelper initialized: base_dir={self._base_dir}")

    def start_run(self) -> str:
        """
        Create new run_XXX directory with auto-increment.

        Returns:
            run_id: Padded numeric ID (e.g., "001", "002")

        Raises:
            TraceError: If cannot create directory
        """
        try:
            # Get next run ID
            run_id = self._get_next_run_id()
            self._current_run_id = run_id

            # Create run directory
            self._current_run_dir = self._base_dir / f"run_{run_id}"
            self._current_run_dir.mkdir(parents=True, exist_ok=True)

            # Initialize metadata
            self._metadata = {
                "run_id": run_id,
                "timestamp_start": datetime.now().isoformat(),
                "timestamp_end": None,
                "total_files": 0,
                "files": []
            }
            self._file_counter = 0

            logging.info(f"🔍 DataTrace: Started run_{run_id} at {self._current_run_dir}")
            return run_id

        except Exception as e:
            raise TraceError(f"Failed to start run: {str(e)}") from e

    def save_df(self, df: pd.DataFrame, name: str) -> str:
        """
        Save DataFrame as CSV with auto-numbering.

        Args:
            df: DataFrame to save
            name: Logical name (e.g., "after_outliers", "OutlierProcessor_input")

        Returns:
            file_path: Full path to saved file (str)

        Raises:
            TraceError: If df is None/invalid or save fails
        """
        if self._current_run_dir is None:
            raise TraceError("No active run. Call start_run() first.")

        if df is None or not isinstance(df, pd.DataFrame):
            raise TraceError(f"Invalid DataFrame for '{name}': {type(df)}")

        try:
            # Format filename with auto-increment
            filename = self._format_filename(name, "csv")
            file_path = self._current_run_dir / filename

            # Save DataFrame
            df.to_csv(file_path, index=True)

            # Collect metadata
            file_metadata = self._collect_df_metadata(df, name, filename)
            self._metadata["files"].append(file_metadata)
            self._metadata["total_files"] += 1

            # Cache last file path
            self._last_file_path = str(file_path)

            logging.debug(f"🔍 Traced: {filename} (shape={df.shape})")
            return str(file_path)

        except Exception as e:
            raise TraceError(f"Failed to save '{name}': {str(e)}") from e

    def save_context(self, context: Dict, name: str = "context") -> str:
        """
        Save context dict as JSON.

        Args:
            context: Dict with pipeline parameters (no DataFrames inside)
            name: Logical name (default: "context")

        Returns:
            file_path: Full path to saved JSON file

        Raises:
            TraceError: If context invalid or save fails
        """
        if self._current_run_dir is None:
            raise TraceError("No active run. Call start_run() first.")

        if not isinstance(context, dict):
            raise TraceError(f"Invalid context for '{name}': {type(context)}")

        try:
            # Format filename with auto-increment
            filename = self._format_filename(name, "json")
            file_path = self._current_run_dir / filename

            # Save context
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(context, f, indent=2, default=str)

            # Collect metadata
            file_metadata = {
                "seq": f"{self._file_counter - 1:02d}",
                "name": name,
                "type": "json",
                "timestamp": datetime.now().isoformat(),
                "size_kb": round(file_path.stat().st_size / 1024, 2)
            }
            self._metadata["files"].append(file_metadata)
            self._metadata["total_files"] += 1

            # Cache last file path
            self._last_file_path = str(file_path)

            logging.debug(f"🔍 Traced: {filename}")
            return str(file_path)

        except Exception as e:
            raise TraceError(f"Failed to save context '{name}': {str(e)}") from e

    def finalize(self) -> None:
        """
        Finalize current run.

        Logic:
            1. Update _metadata.json with timestamp_end
            2. Create/update _latest symlink → current run
            3. Log summary (total files, run duration)

        Note: Safe to call multiple times (idempotent)
        """
        if self._current_run_dir is None:
            logging.warning("No active run to finalize")
            return

        try:
            # Update end timestamp
            self._metadata["timestamp_end"] = datetime.now().isoformat()

            # Write metadata file
            metadata_path = self._current_run_dir / "_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self._metadata, f, indent=2, default=str)

            # Create/update _latest symlink
            latest_link = self._base_dir / "_latest"
            if latest_link.exists() or latest_link.is_symlink():
                latest_link.unlink()

            # Create relative symlink
            try:
                latest_link.symlink_to(f"run_{self._current_run_id}")
            except OSError:
                # Windows might not support symlinks
                logging.warning("Could not create _latest symlink (OS limitation)")

            # Calculate duration
            start_time = datetime.fromisoformat(self._metadata["timestamp_start"])
            end_time = datetime.fromisoformat(self._metadata["timestamp_end"])
            duration = (end_time - start_time).total_seconds()

            logging.info(
                f"🔍 DataTrace: Finalized run_{self._current_run_id} "
                f"({self._metadata['total_files']} files, {duration:.2f}s)"
            )

        except Exception as e:
            logging.error(f"Error finalizing trace: {e}")

    def get_run_dir(self) -> Path:
        """
        Get current run directory Path.

        Returns:
            Path object for active run directory

        Raises:
            TraceError: If no active run (start_run() not called)
        """
        if self._current_run_dir is None:
            raise TraceError("No active run. Call start_run() first.")
        return self._current_run_dir

    def get_last_file_path(self) -> Optional[str]:
        """
        Get path to last saved file (CSV or JSON).

        Returns:
            Full path string or None if no files saved yet
        """
        return self._last_file_path

    # ========== PRIVATE METHODS ==========

    def _get_next_run_id(self) -> str:
        """
        Scan existing run_* directories and return next ID.

        Returns:
            Next run ID as zero-padded string (e.g., "006")
        """
        run_dirs = [d for d in self._base_dir.glob("run_*") if d.is_dir()]
        if not run_dirs:
            return "001"

        # Extract numeric IDs
        max_id = 0
        for d in run_dirs:
            try:
                num_part = d.name.split("_")[1]
                max_id = max(max_id, int(num_part))
            except (IndexError, ValueError):
                continue

        return str(max_id + 1).zfill(3)

    def _format_filename(self, name: str, extension: str) -> str:
        """
        Generate filename with zero-padded counter.

        Args:
            name: Logical name
            extension: "csv" or "json"

        Returns:
            Formatted filename: "02_my_checkpoint.csv"
        """
        filename = f"{self._file_counter:02d}_{name}.{extension}"
        self._file_counter += 1
        return filename

    def _collect_df_metadata(self, df: pd.DataFrame, name: str, filename: str) -> Dict:
        """
        Extract metadata from DataFrame.

        Returns:
            Dict with comprehensive DataFrame metadata
        """
        return {
            "seq": f"{self._file_counter - 1:02d}",
            "name": name,
            "type": "csv",
            "timestamp": datetime.now().isoformat(),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            "null_counts": {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}
        }

    def __str__(self) -> str:
        """String representation for logging."""
        if self._current_run_id:
            return f"DataTraceHelper(run={self._current_run_id}, files={self._metadata['total_files']})"
        return "DataTraceHelper(inactive)"