import json
import logging
from dataclasses import dataclass
from enum import Enum


@dataclass
class PropertyGroupConfig:
    """Property group configuration"""

    name: str
    fields: list
    default_max_age: int
    json_fields: list
    description: str = ""


@dataclass
class TimestampAndTargetFieldNamesConfig:
    """Timestamp and target variable configuration"""

    instrument_type: str
    timestamp_column: str
    target_column: str


# ===== COMMON ENUMERATIONS =====


class FrequencyCategory(Enum):
    """Data frequency categories (common for all modules)."""

    HIGH = "high_frequency"  # 1s-15m
    MEDIUM = "medium_frequency"  # 30m-12h
    LOW = "low_frequency"  # 1d-1M



class DataLengthCategory(Enum):
    """Data length categories (common for all modules)."""

    TINY = "tiny"  # < 20
    SHORT = "short"  # 20-49
    SMALL = "small"  # 50-199
    MEDIUM = "medium"  # 200-999
    LARGE = "large"  # 1000-4999
    HUGE = "huge"  # >= 5000 < 50000
    MASSIVE = "massive"  # >= 50000


def get_frequency_category(interval: str):
    """
    Determine frequency category by interval.

    Args:
        interval: Time series interval

    Returns:
        str: Frequency category ('high_frequency', 'medium_frequency', 'low_frequency')
    """
    high_freq = ["1s", "5s", "15s", "30s", "1m", "5m", "15m"]
    medium_freq = ["30m", "1h", "3h", "6h", "12h"]
    low_freq = ["1d", "3d", "1w", "1M", "3M"]

    if interval in high_freq:
        return FrequencyCategory.HIGH
    elif interval in medium_freq:
        return FrequencyCategory.MEDIUM
    elif interval in low_freq:
        return FrequencyCategory.LOW
    else:
        supported = high_freq + medium_freq + low_freq
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported: {supported}"
        )


def classify_data_length(length: int):
    """
    Data length classification.

    Args:
        length: Time series length

    Returns:
        str: Length category ('tiny', 'short', 'small', 'medium', 'large', 'huge', 'massive')
    """
    if length < 20:
        return DataLengthCategory.TINY
    elif length < 50:
        return DataLengthCategory.SHORT
    elif length < 200:
        return DataLengthCategory.SMALL
    elif length < 1000:
        return DataLengthCategory.MEDIUM
    elif length < 5000:
        return DataLengthCategory.LARGE
    elif length < 50000:
        return DataLengthCategory.HUGE
    else:
        return DataLengthCategory.MASSIVE


class TimeSeriesPreprocessingConfig:
    """Time series preprocessing configuration and metadata"""

    @classmethod
    def __str__(cls):
        return "[timeSeriesProcessing/preprocessingConfig.py][TimeSeriesPreprocessingConfig]"

    # Interval-specific settings for maximum property age (in days)
    MAX_AGE_CONFIG = {
        "1h": {
            "analyzer": 7,  # For hourly data, analyzer properties are valid for 7 days
            "periodicity": 30,  # Periodicity is usually more stable
            "outlier": 3,  # Outlier detection requires more frequent updates
            "outlier_detection": 3,  # Context-aware outlier detection requires frequent updates
            "scaling": 14,  # Scaling properties are moderately stable
            "decomposition": 14,  # Decomposition properties are moderately stable
        },
        "12h": {
            "analyzer": 14,
            "periodicity": 45,
            "outlier": 7,
            "outlier_detection": 5,
            "scaling": 21,
            "decomposition": 21,
        },
        "1d": {
            "analyzer": 30,
            "periodicity": 90,
            "outlier": 14,
            "outlier_detection": 7,
            "scaling": 30,
            "decomposition": 30,
        },
        "3d": {
            "analyzer": 45,
            "periodicity": 120,
            "outlier": 21,
            "outlier_detection": 14,
            "scaling": 45,
            "decomposition": 45,
        },
        "1w": {
            "analyzer": 60,
            "periodicity": 180,
            "outlier": 30,
            "outlier_detection": 21,
            "scaling": 60,
            "decomposition": 60,
        },
        "1M": {
            "analyzer": 90,
            "periodicity": 365,
            "outlier": 45,
            "outlier_detection": 30,
            "scaling": 90,
            "decomposition": 90,
        },
    }

    # Property groups and their definitions
    ANALYZER = PropertyGroupConfig(
        name="analyzer",
        fields=[
            "is_stationary",
            "adf_pvalue",
            "kpss_pvalue",
            "noise_level",
            "estimated_trend_strength",
            "lag1_autocorrelation",
            "rolling_mean_cv",
            "rolling_std_cv",
            "outlier_ratio",
            "volatility",
            "config_analyzer",
            "data_quality_score",
            "missing_ratio",
        ],
        default_max_age=30,
        json_fields=["config_analyzer"],
        description="Properties for time series analysis",
    )

    PERIODICITY = PropertyGroupConfig(
        name="periodicity",
        fields=[
            "main_period",
            "periods",
            "period_confidence_scores",
            "periodicity_detection_method",
            "periodicity_method_results",
            "acf_values",
            "suggested_periods",
            "config_periodicity",
            "detection_status",
            "periodicity_quality_score",
        ],
        default_max_age=60,
        json_fields=[
            "periods",
            "period_confidence_scores",
            "periodicity_method_results",
            "acf_values",
            "suggested_periods",
            "config_periodicity",
        ],
        description="Time series periodicity properties",
    )

    DECOMPOSITION = PropertyGroupConfig(
        name="decomposition",
        fields=[
            "decomposition_method",  # Selected decomposition method
            "trend_strength",  # Trend component strength (from component_strengths)
            "seasonal_strength",  # Seasonal component strength (from component_strengths)
            "residual_strength",  # Residual component strength (from component_strengths)
            "quality_score",  # Overall decomposition quality score
            "reconstruction_error",  # Reconstruction error
            "baseline_quality",  # Baseline method quality (from algorithmDecomposition)
            "model_type",  # Decomposition model type
            "stability_metrics_converged",  # Convergence for MSTL (optional)
            "corrections_applied",  # Applied corrections for MSTL (optional)
            "config_decomposition",  # Full process configuration (JSON)
            # Fourier-specific metrics
            "fourier_n_harmonics",  # Number of significant harmonics
            "fourier_spectral_entropy",  # Spectral entropy
            "fourier_harmonic_strength",  # Total strength of harmonic components
            # SSA-specific metrics
            "ssa_window_length",  # Window length for trajectory matrix
            "ssa_n_components_used",  # Number of components used
            "ssa_variance_explained",  # Fraction of explained variance
            "ssa_component_grouping",  # Component grouping (JSON)
            # N-BEATS-specific metrics
            "nbeats_model_type",  # N-BEATS model type (pytorch_nbeats/linear_fallback)
            "nbeats_training_loss",  # Final training loss
            "nbeats_convergence_achieved",  # Convergence achieved
            "nbeats_harmonic_complexity",  # Harmonic component complexity
            "nbeats_seasonal_harmonics_used",  # Number of harmonics used
            "nbeats_architecture_efficiency",  # Architecture efficiency
        ],
        default_max_age=30,
        json_fields=[
            "corrections_applied",
            "config_decomposition",
            "ssa_component_grouping",
        ],
        description="Extended time series decomposition properties with quality metrics",
    )

    OUTLIER_DETECTION = PropertyGroupConfig(
        name="outlier_detection",
        fields=[
            "contamination_isolation_forest",
            "contamination_lof",
            "n_neighbors_lof",
            "outlier_detection_method",
            "is_combined",
        ],
        default_max_age=15,
        json_fields=[],
        description="Outlier detection properties",
    )

    # OUTLIER_DETECTION = PropertyGroupConfig(
    #     name="outlier_detection",
    #     fields=[
    #         # "outlier_detection_method",
    #         # "ensemble_weights",
    #         # "consensus_threshold",
    #         # "quality_score",
    #         # "confidence_score",
    #         # "financial_regime",
    #         # "config_outlier_detection",
    #         # "processing_status",
    #         # "outlier_indices",
    #         # "severity_scores",
    #         # "component_scores",
    #         # "enhancement_metrics",
    #     ],
    #     default_max_age=7,
    #     json_fields=[
    #         # "ensemble_weights",
    #         # "config_outlier_detection",
    #         # "outlier_indices",
    #         # "severity_scores",
    #         # "component_scores",
    #         # "enhancement_metrics",
    #     ],
    #     description="Extended context-aware outlier detection properties with financial helpers",
    # )

    SCALING = PropertyGroupConfig(
        name="scaling",
        fields=[
            "scaler_type",
            "scaler_params",
            "scaler_serialized",
            "original_stats",
            "scaled_stats",
        ],
        default_max_age=30,
        json_fields=["scaler_params", "original_stats", "scaled_stats"],
        description="Data scaling properties",
    )

    # Timestamp and target variable
    CRYPTO = TimestampAndTargetFieldNamesConfig(
        instrument_type="crypto", timestamp_column="Open_time", target_column="Open"
    )

    # All groups for iteration
    @classmethod
    def all_groups(cls):
        return [
            cls.ANALYZER,
            cls.PERIODICITY,
            cls.OUTLIER_DETECTION,
            cls.SCALING,
            cls.DECOMPOSITION,
        ]

    # Get group by name
    @classmethod
    def get_group_by_name(cls, name):
        for group in cls.all_groups():
            if group.name == name:
                return group
        return None

    # Get max_age settings for specified interval
    @classmethod
    def get_max_age_for_interval(cls, interval):
        """
        Get maximum property age for specified interval

        Args:
            interval: Data interval ('1h', '12h', '1d', '3d', '1w', '1M')

        Returns:
            dict: Dictionary with maximum age for each property group
        """
        if interval in cls.MAX_AGE_CONFIG:
            result = {}
            for group in cls.all_groups():
                try:
                    result[group.name] = cls.MAX_AGE_CONFIG[interval][group.name]
                except KeyError:
                    result[group.name] = group.default_max_age
            return result
        else:
            # If interval not found, use default values
            return {group.name: group.default_max_age for group in cls.all_groups()}

    # Get list of all fields
    @classmethod
    def get_all_fields(cls):
        """
        Get list of all fields for all property groups

        Returns:
            List[str]: List of field names
        """
        fields = []
        for group in cls.all_groups():
            fields.extend(group.fields)
        return fields

    @classmethod
    def normalize_force_recalculate(cls, force_recalculate):
        """
        Normalize force_recalculate parameter to standard format

        Args:
            force_recalculate: bool or Dict - force recalculation settings

        Returns:
            Dict[str, bool]: Dictionary with settings for each group
        """
        if isinstance(force_recalculate, bool):
            return {group.name: force_recalculate for group in cls.all_groups()}
        else:
            result = {}
            for group in cls.all_groups():
                # Check keys in different formats
                if group.name in force_recalculate:
                    result[group.name] = force_recalculate[group.name]
                else:
                    result[group.name] = False
            return result

    @classmethod
    def extract_group_properties(cls, db_props, group_name):
        """
        Extract properties for a specific group from a database record.

        Args:
            db_props: Database property record
            group_name: Name of the property group to extract

        Returns:
            dict: Properties for the requested group
        """
        result = {}

        # Get group configuration
        group = TimeSeriesPreprocessingConfig.get_group_by_name(group_name)
        if not group:
            logging.warning(f"{cls.__str__()}: Unknown property group: {group_name}")
            return result

        # Track extraction statistics
        missing_fields = []

        # Extract each field for this group
        for field in group.fields:
            # Skip fields that don't exist
            if not hasattr(db_props, field):
                missing_fields.append(field)
                continue

            # Get the field value
            value = getattr(db_props, field)

            # Parse JSON fields
            if field in group.json_fields and value is not None:
                try:
                    value = json.loads(value)
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(
                        f"{cls.__str__()}: Failed to parse JSON field {field}: {e}"
                    )

            # Add to result
            result[field] = value

        # Log extraction statistics
        if missing_fields:
            logging.debug(
                f"{cls.__str__()}: Missing fields for {group_name}: {', '.join(missing_fields)}"
            )

        return result