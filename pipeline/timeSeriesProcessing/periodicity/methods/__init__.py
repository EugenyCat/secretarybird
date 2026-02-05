"""
Periodicity detection methods for time series.

Each method implements its own algorithm for detecting periodic patterns
and inherits from BasePeriodicityMethod to ensure a unified interface.
"""

from .basePeriodicityMethod import BasePeriodicityMethod
from .acfMethod import ACFMethod
from .spectralMethod import SpectralMethod
from .waveletMethod import WaveletMethod

__all__ = [
    'BasePeriodicityMethod',
    'ACFMethod',
    'SpectralMethod',
    'WaveletMethod'
]

__version__ = '1.1.0'

# Metadata about available methods
AVAILABLE_METHODS = {
    'acf': {
        'class': ACFMethod,
        'description': 'Autocorrelation Function (ACF) - analysis of series correlation with itself',
        'strengths': ['Simplicity', 'Efficiency for clear patterns', 'Low computational cost'],
        'limitations': ['Sensitivity to non-stationarity', 'Poor performance with noise'],
        'best_for': 'Stationary series with regular periodicity'
    },
    'spectral': {
        'class': SpectralMethod,
        'description': 'Spectral analysis (FFT) - frequency spectrum analysis',
        'strengths': ['Multiple frequencies', 'High resolution', 'Good for harmonics'],
        'limitations': ['Requires stationarity', 'Sensitive to series length'],
        'best_for': 'High-frequency data with clear harmonics'
    },
    'wavelet': {
        'class': WaveletMethod,
        'description': 'Wavelet analysis (CWT) - time-frequency analysis',
        'strengths': ['Works with non-stationary series', 'Local periodicity', 'Noise resilience'],
        'limitations': ['Computational complexity', 'Interpretation difficulty'],
        'best_for': 'Noisy data with variable periods'
    }
}