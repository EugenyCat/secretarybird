"""
Module of time series analysis methods.

Contains concrete implementations of various analysis techniques,
each of which inherits BaseAnalysisMethod.
"""

from .baseAnalysisMethod import BaseAnalysisMethod
from .stationarityMethod import StationarityMethod
from .statisticalMethod import StatisticalMethod
from .outlierAnalysisMethod import OutlierAnalysisMethod

__all__ = [
    'BaseAnalysisMethod',
    'StationarityMethod',
    'StatisticalMethod',
    'OutlierAnalysisMethod'
]

# Registry of available methods for dynamic loading
METHOD_REGISTRY = {
    'stationarity': StationarityMethod,
    'statistical': StatisticalMethod,
    'outlier': OutlierAnalysisMethod
}

# Module version
__version__ = '1.0.0'