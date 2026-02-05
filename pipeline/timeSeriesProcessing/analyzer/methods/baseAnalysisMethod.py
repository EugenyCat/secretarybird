"""
Base class for all time series analysis methods.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional

import pandas as pd

from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.baseModule.baseMethod import BaseTimeSeriesMethod

__version__ = "1.1.0"


class BaseAnalysisMethod(BaseTimeSeriesMethod):
    """
    Base class for all time series analysis methods.

    Inherits common functionality from BaseTimeSeriesMethod.
    Defines specific interface and functionality for analysis methods.
    """

    # Standard default configurations for analysis
    DEFAULT_CONFIG = {
        **BaseTimeSeriesMethod.DEFAULT_CONFIG,
        # "min_data_length": 3,       # adapted in configAnalyzer
        # "max_missing_ratio": 0.5,   # adapted in configAnalyzer
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize analysis method.

        Args:
            config: Method configuration
        """
        # Merge configuration with analysis defaults
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

        # Validate basic analysis parameters
        validate_required_locals(["min_data_length"], self.config)

    @abstractmethod
    def process(
        self, data: pd.Series, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute time series analysis.

        Args:
            data: Time series for analysis
            context: Additional context (interval, parameters, etc.)

        Returns:
            Dict with standard format:
            {
                'status': 'success/error',
                'result': {...},     # analysis results
                'metadata': {...}    # metadata
            }
        """
        pass

    def validate_input(
        self, data: pd.Series, min_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extended validation of input data for analysis.

        Args:
            data: Time series for validation
            min_length: Minimum data length (default from config)

        Returns:
            Dict with validation result
        """
        # Use basic validation
        min_len = min_length or self.config["min_data_length"]
        return super().validate_input(data, min_len)