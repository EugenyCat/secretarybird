"""
OutlierDetection module for time series processing.

Provides production-ready outlier detection and removal functionality
with Enhanced Decision Tree, tiered execution strategy, and 95%+ context reuse.

Architecture:
- Level 1: OutlierDetectionProcessor (BaseProcessor inheritance)
- Level 2: OutlierDetectionAlgorithm (Enhanced Decision Tree)
- Level 3: Detection Methods (Tier 1-4 strategies)

Key Features:
- Statistical Enhancement Pattern (zero-cost detection)
- Component-Aware Processing (decomposition reuse)
- Financial Helpers Integration (regime-aware detection)
- Tiered Detection Strategy (early stopping optimization)
- Production-Ready Framework (validation, monitoring, graceful degradation)
"""

__version__ = "1.0.0"
__author__ = "Time Series Processing Team"

from pipeline.timeSeriesProcessing.outlierDetection.algorithmOutlierDetection import (
    OutlierDetectionAlgorithm,
)
from pipeline.timeSeriesProcessing.outlierDetection.configOutlierDetection import (
    OutlierDetectionConfigAdapter,
    build_config_from_properties,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.baseOutlierDetectionMethod import (
    BaseOutlierDetectionMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.componentAnomalyMethod import (
    ComponentAnomalyMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.methods.statisticalEnhancementMethod import (
    StatisticalEnhancementMethod,
)
from pipeline.timeSeriesProcessing.outlierDetection.processorOutlierDetection import (
    OutlierDetectionProcessor,
)

__all__ = [
    "BaseOutlierDetectionMethod",
    "StatisticalEnhancementMethod",
    "ComponentAnomalyMethod",
    "OutlierDetectionConfigAdapter",
    "build_config_from_properties",
    "OutlierDetectionAlgorithm",
    "OutlierDetectionProcessor",
]
