"""
Detection methods for outlierDetection module.

Contains base class and specific detection methods implementing
various outlier detection strategies.
"""

__version__ = "1.0.0"

from pipeline.timeSeriesProcessing.outlierDetection.methods.baseOutlierDetectionMethod import (
    BaseOutlierDetectionMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.componentAnomalyMethod import (
    ComponentAnomalyMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.statisticalEnhancementMethod import (
    StatisticalEnhancementMethod,
)

__all__ = [
    "BaseOutlierDetectionMethod",
    "StatisticalEnhancementMethod",
    "ComponentAnomalyMethod",
]
