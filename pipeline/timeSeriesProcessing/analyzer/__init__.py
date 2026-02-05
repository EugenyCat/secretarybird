"""
Time series analysis module.

Provides comprehensive time series analysis through a system
of modular methods managed by an orchestrator and processor.

Components:
- AnalysisProcessor: Lifecycle management and pipeline integration
- TimeSeriesAnalyzer: Analysis method orchestration
- methods/: Concrete analysis method implementations
"""

from .processorAnalyzer import AnalysisProcessor
from .algorithmAnalyzer import TimeSeriesAnalyzer

__all__ = [
    'AnalysisProcessor',
    'TimeSeriesAnalyzer'
]

# Module version
__version__ = '1.0.0'