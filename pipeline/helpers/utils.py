import base64
import logging
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler

from pipeline.helpers.setup import ConfigurationBuilder


class ETLUtils(ConfigurationBuilder):
    """
    Utility class for various ETL helper functions.
    """

    @staticmethod
    def count_months(start_date, end_date):
        """
        Calculate the number of months between two dates (static method).
        """
        return (
            (end_date.year - start_date.year) * 12
            + end_date.month
            - start_date.month
            + 1
        )

    @staticmethod
    def convert_intervals_into_ms(intervals):
        """
        Converts a list of datetime intervals into milliseconds since the epoch (static method)..

        Args:
            intervals (list of tuples): A list of tuples where each tuple contains
                                        two datetime objects (interval_start, interval_end).

        Returns:
            list of tuples: A list of tuples where each datetime object is converted
                            to milliseconds since the epoch.
        """
        return [
            tuple(int(x.timestamp() * 1000) for x in el_interval)
            for el_interval in intervals
        ]

    def get_list_of_intervals(self):
        """
        Splits a large date range into smaller intervals, ensuring that each interval
        contains no more than 500 records, to comply with Binance API limits.

        This function divides the specified time range into multiple intervals, each
        of which includes up to 500 data points, making it suitable for handling
        requests to the Binance API which has a maximum limit on the number of records
        returned per request.

        Args:
            start (datetime): The start date of the overall interval.
            end (datetime): The end date of the overall interval.

        Returns:
            list of tuples: A list where each tuple represents an interval of the form
                            (interval_start, interval_end), with intervals designed
                            to fit within the 500-record limit of the API.
        """

        total_days = (self.end - self.start).days
        total_years = self.end.year - self.start.year

        # when date range is small and doesn't exceed 500 data points
        if (
            (self.interval == "1h" and total_days < 19)
            or (self.interval == "12h" and total_days < 250)
            or (self.interval == "1d" and total_days < 500)
            or (self.interval == "3d" and total_days < 1500)
            or (self.interval == "1w" and total_days < 3500)
            or (self.interval == "1M" and total_years < 40)
        ):
            inters = [(self.start, self.end)]
        else:  # when date range is big and exceeds 500 data points
            if self.interval == "12h":
                rel_data = relativedelta(months=7) + relativedelta(day=31)
            elif self.interval == "1d":
                rel_data = (
                    relativedelta(years=1)
                    + relativedelta(months=3)
                    + relativedelta(day=31)
                )
            elif self.interval == "3d":
                rel_data = (
                    relativedelta(years=3)
                    + relativedelta(months=9)
                    + relativedelta(day=31)
                )
            elif self.interval == "1w":
                rel_data = relativedelta(years=9) + relativedelta(day=31)
            elif self.interval == "1M":
                rel_data = relativedelta(years=40) + relativedelta(day=31)
            else:  # elif interval == '1h'
                rel_data = relativedelta(days=19)

            inters = []

            # Calculate tuples represent an interval (interval_start, interval_end)
            current_date = self.start
            while current_date < self.end - rel_data:
                inters.append(
                    (
                        current_date,
                        (current_date + rel_data).replace(
                            hour=23, minute=59, second=59
                        ),
                    )
                )
                current_date = (
                    current_date + rel_data + relativedelta(days=1)
                ).replace(hour=0, minute=0, second=0)

            # Add the last interval
            inters.append((current_date, self.end))

        return inters


class MLUtils(ConfigurationBuilder):

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str = "Close",
        look_back: int = 30,
        test_size: float = 0.2,
    ):
        """
        Prepares time series data for model training by performing preprocessing steps such as:
        - Converting timestamps
        - Creating lag features
        - Scaling features
        - Splitting into train and test sets

        Args:
            df (pd.DataFrame): Input dataframe containing time series data.
            target_column (str): The column to predict (default is 'Close').
            look_back (int): Number of previous time steps to use as input features (default is 30).
            test_size (float): Proportion of data to use for testing (default is 0.2).

        Returns:
            dict: A dictionary containing the prepared train and test datasets.
        """

        # Ensure that 'Open time' is in datetime format
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")

        # Sort data by timestamp (ascending)
        df = df.sort_values(by="Open time")

        # Create lag features (use previous 'look_back' values to predict the current value)
        for i in range(1, look_back + 1):
            df[f"lag_{i}"] = df[target_column].shift(i)

        # Drop rows with NaN values generated by lagging
        df.dropna(inplace=True)

        # Feature columns (exclude 'Open time' and target column)
        feature_columns = [
            col for col in df.columns if col not in ["Open time", target_column]
        ]

        # Define X (features) and y (target)
        X = df[feature_columns].values
        y = df[target_column].values

        # Scale features using MinMaxScaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and test sets (for time series, we split by time)
        n_train = int(
            (1 - test_size) * len(df)
        )  # Index where the train-test split occurs
        X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

        # Return the prepared datasets
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler,
        }


# Improved serialization/deserialization functions for utils.py
def serialize_object(obj: Any):
    """Improved object serialization to base64 string"""
    if obj is None:
        logging.warning("Attempting to serialize None object")
        return None

    try:
        serialized = base64.b64encode(pickle.dumps(obj)).decode("utf-8")
        # Check serialization success
        if not serialized:
            logging.warning("Serialization returned empty string")
            return None
        return serialized
    except Exception as e:
        logging.warning(f"Object serialization error: {str(e)}")
        return None


def deserialize_object(serialized: str):
    """Improved object deserialization from base64 string"""
    if not serialized:
        logging.warning("Attempting to deserialize empty string")
        return None

    try:
        return pickle.loads(base64.b64decode(serialized))
    except Exception as e:
        logging.warning(f"Object deserialization error: {str(e)}")
        return None


def to_json_safe(data):
    """
    Convert data to JSON-compatible format

    Args:
        data: Data to convert

    Returns:
        dict or list with JSON-compatible data
    """
    if isinstance(data, (str, int, float, bool, type(None))):
        return data
    elif isinstance(data, (list, tuple)):
        return [to_json_safe(item) for item in data]
    elif isinstance(data, dict):
        return {str(k): to_json_safe(v) for k, v in data.items()}
    else:
        # For objects that cannot be directly serialized
        return str(data)


def should_use_direct_method(data_length, interval=None):
    """
    Determines whether to use direct outlier detection method instead of decomposition

    This method uses adaptive thresholds depending on the time series interval.
    For short series or series with certain intervals (e.g., monthly)
    direct method without decomposition is recommended:

    - For monthly data (1M, 3M) direct method is always used
    - For high-frequency data (1s, 1min) high thresholds are used (7200, 1440)
    - For low-frequency data (1h, 1d) low thresholds are used (168, 60)
    - By default, a dynamic threshold of 20% of series length is used (minimum 60 points)

    Args:
        data_length (int): Number of points in the time series
        interval (str, optional): Data interval ('1s', '1m', '1h', '1d', '1M', etc.)

    Returns:
        bool: True if direct method should be used (without decomposition),
              False if combined method with decomposition should be used
    """
    # For monthly data always use direct method
    if interval in ["1M", "3M"]:
        return True

    # Adaptive thresholds depending on interval
    if interval == "1s":
        return data_length < 7200  # 2 hours for second data
    elif interval == "1m":
        return data_length < 1440  # 1 day for minute data
    elif interval == "5m":
        return data_length < 576  # 2 days for 5-minute data
    elif interval == "15m":
        return data_length < 384  # 4 days for 15-minute data
    elif interval == "1h":
        return data_length < 168  # 1 week for hourly data
    elif interval == "1d":
        return data_length < 60  # 2 months for daily data

    # By default - determine threshold dynamically
    # Minimum 60 points for decomposition or 20% of total volume
    min_threshold = max(60, int(data_length * 0.2))
    return data_length < min_threshold


def get_interval_thresholds(interval):
    """
    Determines adaptive data size thresholds based on interval

    This method returns three thresholds used for making decisions
    about outlier detection method selection and processing strategy:

    - small_threshold: threshold for determining small datasets
    - medium_threshold: threshold for medium datasets
    - large_threshold: threshold for large datasets

    Thresholds scale depending on interval - for high-frequency data
    (1s, 1min) thresholds are higher, and for low-frequency (1d, 1w, 1M) thresholds are lower.

    These thresholds are used in _select_optimal_method to determine optimal
    outlier detection algorithm (LOF or Isolation Forest).

    Args:
        interval (str): Data interval ('1s', '15s', '1m', '1h', '1d', '1w', '1M', etc.)

    Returns:
        tuple: (small_threshold, medium_threshold, large_threshold) - thresholds for
               dataset size classification
    """
    # Determine thresholds depending on interval
    if any(x in interval for x in ["1s", "15s", "30s"]):
        return 180, 3600, 14400  # 3 minutes, 1 hour, 4 hours for second data
    elif any(x in interval for x in ["1m", "5m"]):
        return 60, 720, 2880  # 1 hour, 12 hours, 2 days for minute data
    elif any(x in interval for x in ["15m", "30m"]):
        return 30, 192, 960  # Several hours, 2 days, 10 days for 15-minute
    elif any(x in interval for x in ["1h", "3h", "12h"]):
        return 30, 100, 500  # 1-3 days, ~4 days, ~3 weeks for hourly
    elif "1d" in interval or "3d" in interval:
        return 30, 90, 365  # 1 month, 3 months, 1 year for daily
    elif "1w" in interval:
        return 12, 26, 260  # 3 months, 6 months, 5 years for weekly
    elif "1M" in interval:
        return 12, 36, 120  # 1 year, 3 years, 10 years for monthly
    else:
        # For unknown intervals - general thresholds
        return 30, 100, 500


def estimate_n_neighbors(data_len, interval):
    """
    Adaptive estimation of optimal number of neighbors for LOF algorithm

    Calculates optimal number of neighbors (n_neighbors) for Local Outlier Factor
    algorithm based on data size and interval. Takes into account:

    1. Base scaling by series length:
       - For very short series (<20): n_neighbors = 25% of data size
       - For small series (<100): n_neighbors = 15% of data size
       - For medium series (<1000): n_neighbors = 5% of data size
       - For large series (≥1000): logarithmic growth of neighbor count

    2. Adjustment based on interval:
       - For high-frequency data (1s, 1min): increase n_neighbors by 1.5-2 times
       - For low-frequency data (1d, 1M): decrease n_neighbors by 0.5-0.7 times

    Larger number of neighbors makes LOF less sensitive to local outliers,
    while smaller number increases sensitivity but may lead to false
    positives.

    Args:
        data_len (int): Time series length
        interval (str, optional): Data interval ('1s', '1min', '1h', '1d', '1M', etc.)

    Returns:
        int: Estimated optimal number of neighbors for LOF
    """
    # Base estimate by length
    if data_len < 20:
        base_neighbors = min(5, max(2, int(data_len * 0.25)))
    elif data_len < 100:
        base_neighbors = min(10, max(3, int(data_len * 0.15)))
    elif data_len < 1000:
        base_neighbors = min(30, max(10, int(data_len * 0.05)))
    else:
        # Logarithmic growth for very long series
        base_neighbors = min(100, max(30, int(10 * np.log10(data_len))))

    # Adjustment for data interval
    if interval:
        if interval == "1s":
            # For second data increase number of neighbors
            multiplier = 2.0
        elif interval == "1min":
            multiplier = 1.5
        elif interval == "1h":
            multiplier = 1.0
        elif interval == "1d":
            # For daily data decrease number of neighbors
            multiplier = 0.7
        elif interval in ["1M", "3M"]:
            multiplier = 0.5
        else:
            multiplier = 1.0

        adjusted_neighbors = int(base_neighbors * multiplier)
        # Ensure minimum 2 neighbors and no more than half of series length
        return min(data_len // 2, max(2, adjusted_neighbors))

    return min(data_len // 2, max(2, base_neighbors))


def get_optimal_isolation_forest_params(data_len, interval):
    """
    Calculates optimal parameters for Isolation Forest algorithm

    Determines two key parameters for Isolation Forest depending on
    data size and interval:

    1. n_estimators: number of trees in ensemble
       - For small datasets (<1000): from 50 to 100 trees
       - For large datasets (≥1000): logarithmically grows from 100 to 300

    2. max_samples: maximum number of samples for training one tree
       - Scales depending on interval - for high-frequency data
         (seconds, minutes) more samples are used
       - For low-frequency data (days, weeks, months) fewer samples are used
       - Base value changes from 512 to 1024 depending on interval
       - Also accounts for percentage of total data size (from 5% to 15%)

    Increasing n_estimators improves accuracy but requires more resources.
    Optimal max_samples affects model's ability to detect
    local versus global outliers.

    Args:
        data_len (int): Time series length
        interval (str): Data interval ('1s', '15s', '1m', '1h', '1d', '1w', '1M', etc.)

    Returns:
        tuple: (n_estimators, max_samples) - optimal parameters for Isolation Forest

    Examples:
        >>> get_optimal_isolation_forest_params(500, '1h')
        (50, 128) # 50 trees, 128 samples for hourly series of length 500
        >>> get_optimal_isolation_forest_params(10000, '1s')
        (170, 1024) # 170 trees, 1024 samples for second series of length 10000
    """
    """Optimized Isolation Forest parameters accounting for wide range of intervals"""

    # Base scaling of n_estimators with logarithmic growth
    if data_len < 1000:
        n_estimators = min(100, max(50, int(data_len / 20)))
    else:
        n_estimators = min(300, max(100, int(50 * np.log10(data_len / 1000 + 1) + 100)))

    # Determine interval factor (from 0 to 1)
    # 0 = shortest (seconds), 1 = longest (month)
    interval_factor = 0.5  # default value

    if any(x in interval for x in ["1s", "15s", "30s"]):
        interval_factor = 0.0  # seconds
    elif any(x in interval for x in ["1m", "5m"]):
        interval_factor = 0.2  # short minutes
    elif any(x in interval for x in ["15m", "30m"]):
        interval_factor = 0.3  # long minutes
    elif any(x in interval for x in ["1h", "3h"]):
        interval_factor = 0.5  # short hours
    elif any(x in interval for x in ["12h", "1d"]):
        interval_factor = 0.7  # day
    elif any(x in interval for x in ["3d", "1w"]):
        interval_factor = 0.8  # week
    elif "1M" in interval:
        interval_factor = 1.0  # month

    # Smooth scaling of max_samples depending on interval_factor
    # The smaller the interval, the more points needed (for high-frequency data)
    max_samples_base = int(1024 * (1 - interval_factor * 0.5))  # from 1024 to 512
    max_samples_ratio = 0.05 + interval_factor * 0.1  # from 5% to 15%

    max_samples = min(max_samples_base, max(128, int(data_len * max_samples_ratio)))

    return n_estimators, max_samples


def get_analyzer_property(context, property_name, default_value=None):
    """
    Get analyzer property from context

    Args:
        context: Processing context
        property_name: Property name
        default_value: Default value

    Returns:
        Property value or default_value
    """
    try:
        # Explicit check of each level without using .get()
        try:
            current_properties = context["currentProperties"]
        except KeyError:
            logging.debug(
                f"CurrentProperties not found, using default {property_name}={default_value}"
            )
            return default_value

        try:
            analyzer_props = current_properties["analyzer"]
        except KeyError:
            logging.debug(
                f"Analyzer properties not found, using default {property_name}={default_value}"
            )
            return default_value

        try:
            value = analyzer_props[property_name]
        except KeyError:
            logging.debug(
                f"Property {property_name} not found, using default {property_name}={default_value}"
            )
            return default_value

        if value is not None:
            logging.debug(f"Retrieved {property_name}={value} from context")
            return value
        else:
            logging.debug(
                f"Property {property_name} is None, using default {property_name}={default_value}"
            )
            return default_value

    except Exception as e:
        logging.warning(f"Error retrieving {property_name} from context: {e}")
        return default_value


def get_periodicity_property(context, property_name: str, default_value=None):
    """
    Get periodicity property from context

    Args:
        context: Processing context
        property_name: Property name
        default_value: Default value

    Returns:
        Property value or default_value
    """
    try:
        # Explicit check of each level without using .get()
        try:
            current_properties = context["currentProperties"]
        except KeyError:
            logging.debug(
                f"CurrentProperties not found, using default periodicity {property_name}={default_value}"
            )
            return default_value

        try:
            periodicity_props = current_properties["periodicity"]
        except KeyError:
            logging.debug(
                f"Periodicity properties not found, using default {property_name}={default_value}"
            )
            return default_value

        try:
            value = periodicity_props[property_name]
        except KeyError:
            logging.debug(
                f"Periodicity property {property_name} not found, using default {property_name}={default_value}"
            )
            return default_value

        if value is not None:
            logging.debug(f"Retrieved periodicity {property_name}={value} from context")
            return value
        else:
            logging.debug(
                f"Periodicity property {property_name} is None, using default {property_name}={default_value}"
            )
            return default_value

    except Exception as e:
        logging.warning(
            f"Error retrieving periodicity {property_name} from context: {e}"
        )
        return default_value


def get_outlier_property(context, property_name, default_value=None):
    """
    Get outlier property from context

    Args:
        context: Processing context
        property_name: Property name
        default_value: Default value

    Returns:
        Property value or default_value
    """
    try:
        # Explicit check of each level without using .get()
        try:
            current_properties = context["currentProperties"]
        except KeyError:
            logging.debug(
                f"CurrentProperties not found, using default outlier {property_name}={default_value}"
            )
            return default_value

        try:
            outlier_props = current_properties["outlier"]
        except KeyError:
            logging.debug(
                f"Outlier properties not found, using default {property_name}={default_value}"
            )
            return default_value

        try:
            value = outlier_props[property_name]
        except KeyError:
            logging.debug(
                f"Outlier property {property_name} not found, using default {property_name}={default_value}"
            )
            return default_value

        if value is not None:
            logging.debug(f"Retrieved outlier {property_name}={value} from context")
            return value
        else:
            logging.debug(
                f"Outlier property {property_name} is None, using default {property_name}={default_value}"
            )
            return default_value

    except Exception as e:
        logging.warning(f"Error retrieving outlier {property_name} from context: {e}")
        return default_value


def get_all_characteristics(context):
    """
    Get all time series characteristics from context

    Args:
        context: Processing context

    Returns:
        Dictionary with characteristics
    """
    characteristics = {}

    # Get all analyzer properties with explicit check
    try:
        current_properties = context["currentProperties"]
    except KeyError:
        logging.debug("CurrentProperties not found, returning empty characteristics")
        return characteristics

    try:
        analyzer_props = current_properties["analyzer"]
    except KeyError:
        analyzer_props = {}
        logging.debug("Analyzer properties not found, using empty dict")

    # Safe extraction of base characteristics without .get()
    def safe_extract(props_dict, key, default):
        try:
            return props_dict[key]
        except KeyError:
            return default

    characteristics["length"] = safe_extract(analyzer_props, "length", 0)
    characteristics["missing_ratio"] = safe_extract(
        analyzer_props, "missing_ratio", 0.0
    )
    characteristics["volatility"] = safe_extract(analyzer_props, "volatility", 0.0)
    characteristics["trend_strength"] = safe_extract(
        analyzer_props, "trend_strength", 0.0
    )
    characteristics["noise_level"] = safe_extract(analyzer_props, "noise_level", 0.0)
    characteristics["autocorrelation"] = safe_extract(
        analyzer_props, "autocorrelation", 0.0
    )
    characteristics["skewness"] = safe_extract(analyzer_props, "skewness", 0.0)
    characteristics["kurtosis"] = safe_extract(analyzer_props, "kurtosis", 0.0)

    # Stationarity
    characteristics["is_stationary"] = safe_extract(
        analyzer_props, "is_stationary", False
    )
    characteristics["stationarity"] = safe_extract(analyzer_props, "stationary", False)

    # Outliers - get from different sources
    try:
        outlier_props = current_properties["outlier"]
    except KeyError:
        outlier_props = {}
        logging.debug("Outlier properties not found, using empty dict")

    analyzer_outlier_ratio = safe_extract(analyzer_props, "outlier_ratio", 0.0)
    outlier_outlier_ratio = safe_extract(outlier_props, "outlier_ratio", 0.0)
    characteristics["outlier_ratio"] = max(
        analyzer_outlier_ratio, outlier_outlier_ratio
    )

    # Period
    try:
        periodicity_props = current_properties["periodicity"]
    except KeyError:
        periodicity_props = {}
        logging.debug("Periodicity properties not found, using empty dict")

    characteristics["main_period"] = safe_extract(periodicity_props, "main_period", 0)
    characteristics["periods"] = safe_extract(periodicity_props, "periods", [])
    characteristics["stl_trend_strength"] = safe_extract(
        periodicity_props, "stl_trend_strength", 0.0
    )
    characteristics["stl_seasonal_strength"] = safe_extract(
        periodicity_props, "stl_seasonal_strength", 0.0
    )

    logging.debug(f"Retrieved characteristics: {list(characteristics.keys())}")
    return characteristics


def check_required_properties(context, required_groups):
    """
    Check presence of required property groups in context

    Args:
        context: Processing context
        required_groups: List of required groups ('analyzer', 'periodicity', 'outlier')

    Returns:
        Dictionary with check results for each group
    """
    results = {}

    # Explicit check of currentProperties without .get()
    try:
        current_properties = context["currentProperties"]
    except KeyError:
        logging.warning("CurrentProperties not found in context")
        # All groups are missing
        return {group: False for group in required_groups}

    for group in required_groups:
        results[group] = group in current_properties and bool(current_properties[group])

    missing_groups = [group for group, present in results.items() if not present]
    if missing_groups:
        logging.warning(f"Missing property groups in context: {missing_groups}")

    return results


def get_interval(context):
    """Get interval from context"""
    try:
        return context["interval"]
    except KeyError:
        logging.debug("Interval not found in context")
        return None


def calculate_volatility(data: pd.Series) -> float:
    """
    Calculate time series volatility

    Args:
        data: Time series for analysis

    Returns:
        float: Volatility value (coefficient of variation)
    """
    if len(data) < 2 or data.std() == 0:
        return 0.0
    return data.std() / abs(data.mean()) if data.mean() != 0 else data.std()


def estimate_trend_strength(data: pd.Series) -> float:
    """
    Estimate trend strength through linear regression

    Calculates trend strength as a combination of line slope and quality
    of fit (r^2), normalized in range [0, 1].

    Args:
        data: Time series for analysis

    Returns:
        float: Trend strength in range from 0 (no trend) to 1 (strong trend)
    """
    if len(data) < 3:
        return 0.0

    try:
        x = np.arange(len(data))
        slope, _, r_value, _, _ = linregress(x, data.values)

        # Normalization by mean value
        normalized_slope = abs(slope) / (abs(data.mean()) + 1e-10)
        trend_strength = min(1.0, normalized_slope * 100) * (r_value**2)

        return max(0.0, min(1.0, trend_strength))
    except Exception as e:
        logging.warning(f"Error calculating trend strength: {e}")
        return 0.0


def estimate_noise_level(data: pd.Series) -> float:
    """
    Estimate noise level through second-order differences

    Calculates relative noise level as the ratio of standard deviation
    of second-order differences to standard deviation of data.

    Args:
        data: Time series for analysis

    Returns:
        float: Noise level in range from 0 (no noise) to 1 (very noisy series)
    """
    if len(data) < 3:
        return 0.0

    try:
        # Estimate through second-order differences
        diff2 = data.diff().diff().dropna()
        if len(diff2) == 0:
            return 0.0

        noise_std = diff2.std()
        data_std = data.std()

        if data_std == 0:
            return 0.0

        noise_level = noise_std / data_std
        return min(1.0, max(0.0, noise_level))
    except Exception as e:
        logging.warning(f"Error calculating noise level: {e}")
        return 0.0


def validate_required_locals(required_params: list, input_params: dict):
    """
    Validation through locals() - the most efficient way

    Usage:
        validate_required_locals(['ts_id', 'currency'], locals())
    """
    missing_params = [
        param
        for param in required_params
        if param not in input_params or input_params[param] is None
    ]
    if missing_params:
        raise ValueError(f"Required parameters missing: {missing_params}")