"""
Financial helpers for OutlierDetection module.

Modules:
- regime: Market regime classification
- alpha: Alpha potential assessment (in development)
- microstructure: Microstructure analysis (in development)
- correlation: Cross-asset correlation analysis (in development)
"""

from .regime import (
    calculate_regime_features,
    classify_market_regime,
    get_regime_thresholds,
)

__version__ = "1.0.0"

__all__ = [
    "classify_market_regime",
    "get_regime_thresholds",
    "calculate_regime_features",
]