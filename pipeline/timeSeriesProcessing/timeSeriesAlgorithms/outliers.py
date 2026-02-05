import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pipeline.timeSeriesProcessing.timeSeriesAlgorithms.decomposition import TrendDecomposer
from pipeline.helpers.utils import (
    should_use_direct_method,
    get_interval_thresholds,
    estimate_n_neighbors,
    get_optimal_isolation_forest_params
)


class OutlierDetector:
    """Optimized class for detecting and removing time series outliers"""

    def __init__(self, interval, max_outlier_ratio=0.3):
        """
        Initialize outlier detector

        Args:
            interval: Data interval (e.g., '1s', '1min', '1h', '1d')
            max_outlier_ratio: Maximum outlier ratio (0.0-1.0)
        """
        self.interval = interval
        self._cache = {}
        self._metadata = {}
        self.max_outlier_ratio = max_outlier_ratio

        self._trend_decomposer = None  # Caching decomposer
        self.reset_metadata()

    def reset_metadata(self):
        """Reset outlier detection metadata"""
        self._metadata = {
            "outlier_ratio": 0.0,
            "contamination_isolation_forest": None,
            "contamination_lof": None,
            "n_neighbors_lof": None,
            "outlier_detection_method": None,
            "is_combined": 0,
        }

    def get_metadata(self):
        """Get metadata from the last outlier detection operation"""
        return self._metadata.copy()

    def _process_with_method(
            self, data, method, contamination=None, n_neighbors=None, is_residual=False
    ):
        """
        Generalized method for processing outliers with various algorithms

        Args:
            data: Time series
            method: 'isolation_forest' or 'lof'
            contamination: Outlier ratio
            n_neighbors: Number of neighbors for LOF
            is_residual: True if processing residual component

        Returns:
            pd.Series: Cleaned series
        """
        if data is None or len(data) < 5:
            return data

        # Define parameters
        if contamination is None:
            contamination = self._optimize_contamination(data, method)

        if method == "lof" and n_neighbors is None:
            n_neighbors = estimate_n_neighbors(len(data), self.interval)
            self._metadata["n_neighbors_lof"] = n_neighbors

        # Apply algorithm
        if method == "isolation_forest":
            result = self._apply_isolation_forest(data, contamination)
            self._metadata["contamination_isolation_forest"] = contamination
            self._metadata["outlier_detection_method"] = "isolation_forest"
        elif method == "lof":
            result = self._apply_lof(data, contamination, n_neighbors)
            self._metadata["contamination_lof"] = contamination
            self._metadata["outlier_detection_method"] = "lof"
        else:
            raise ValueError(f"Unsupported method: {method}")

        # For the final series (not residual component) calculate outlier_ratio
        if not is_residual:
            self._calculate_outlier_ratio(data, result)

        return result

    def _calculate_outlier_ratio(self, original_data, cleaned_data):
        """Calculate outlier ratio based on data changes"""
        # Convert to numpy
        orig_values = (
            original_data.values
            if hasattr(original_data, "values")
            else np.array(original_data)
        )
        clean_values = (
            cleaned_data.values
            if hasattr(cleaned_data, "values")
            else np.array(cleaned_data)
        )

        # Convert to 1D array
        orig_values = orig_values.flatten()
        clean_values = clean_values.flatten()

        # Calculate changed points
        changed_points = np.sum(orig_values != clean_values)
        total_points = len(orig_values)

        # Calculate ratio
        ratio = changed_points / total_points if total_points > 0 else 0.0

        # Check for exceeding expected value
        expected = max(
            self._metadata.get("contamination_isolation_forest", 0.0) or 0.0,
            self._metadata.get("contamination_lof", 0.0) or 0.0,
        )

        # If deviation from expected value is more than 30%, log warning
        if ratio > 0 and expected > 0 and (ratio / expected) > 1.3:
            logging.warning(
                f"Significant deviation of outlier_ratio ({ratio:.4f}) from expected contamination ({expected:.4f}). "
                f"Ratio: {ratio / expected:.2f}. Recommended to use direct method (is_combined=0)."
            )

        self._metadata["outlier_ratio"] = float(ratio)
        return ratio

    def _limit_outlier_ratio(self, data, outlier_mask):
        """Limits the outlier ratio to maximum value"""
        outlier_count = np.sum(outlier_mask)
        total_count = len(outlier_mask)
        current_ratio = outlier_count / total_count if total_count > 0 else 0

        if current_ratio <= self.max_outlier_ratio:
            return outlier_mask

        # Determine the most extreme outliers
        values = (
            data.values.flatten()
            if hasattr(data, "values")
            else np.array(data).flatten()
        )
        non_outlier_mask = ~outlier_mask

        # Calculate distribution center (median of non-anomalous points)
        center = (
            np.median(values[non_outlier_mask])
            if np.any(non_outlier_mask)
            else np.median(values)
        )
        deviation = np.abs(values - center)

        # Sort outliers by deviation degree
        outlier_indices = np.where(outlier_mask)[0]
        if len(outlier_indices) == 0:
            return outlier_mask

        outlier_deviations = deviation[outlier_indices]
        sorted_indices = outlier_indices[np.argsort(-outlier_deviations)]

        # Keep only the most extreme
        max_outliers = max(1, int(self.max_outlier_ratio * total_count))
        max_outliers = min(max_outliers, len(sorted_indices))

        limited_mask = np.zeros_like(outlier_mask, dtype=bool)
        limited_mask[sorted_indices[:max_outliers]] = True

        return limited_mask

    def _optimize_contamination(self, data, method="isolation_forest"):
        """Optimize contamination value for the algorithm"""
        if data is None or len(data) == 0:
            return 0.1

        data_len = len(data)

        # For small datasets use a more conservative approach
        if data_len < 30:
            estimated_contamination = min(0.05, 2 / data_len)
            return estimated_contamination

        # Use cache for acceleration
        data_hash = hash(
            str(data.shape) + str(data.iloc[::10].sum())
            if hasattr(data, "iloc")
            else str(len(data)) + str(np.sum(data[::10]))
        )
        cache_key = (data_hash, method)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Quick contamination estimate based on IQR
        try:
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            estimated_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            estimated_contamination = max(estimated_outliers / len(data), 0.001)
        except Exception:
            estimated_contamination = 0.05  # Default value

        # Limit the value
        estimated_contamination = min(
            self.max_outlier_ratio, max(estimated_contamination, 0.001)
        )

        # For small datasets or when estimate is accurate enough, avoid complex optimization
        if data_len < 100 or estimated_outliers > 0:
            self._cache[cache_key] = estimated_contamination
            return estimated_contamination

        # For more complex cases use parallel search
        try:
            # Define search boundaries
            min_cont = max(
                0.001, estimated_contamination * 0.3
            )  # or max(0.001, estimated_contamination * 0.5)
            max_cont = min(
                self.max_outlier_ratio, min(0.4, estimated_contamination * 2.0)
            )  # or min(self.max_outlier_ratio, estimated_contamination * 1.5)

            # Create adaptive grid with 6 points
            grid_size = 6
            contamination_range = np.linspace(min_cont, max_cont, grid_size).tolist()

            # Guarantee that original estimate is exactly included in grid
            if estimated_contamination not in contamination_range:
                contamination_range.append(estimated_contamination)
                contamination_range.sort()

            results = Parallel(n_jobs=2)(
                delayed(self._evaluate_contamination)(data, cont, method)
                for cont in contamination_range
            )
            best_contamination, _ = min(results, key=lambda x: x[1])
            best_contamination = min(best_contamination, self.max_outlier_ratio)
        except Exception:
            best_contamination = estimated_contamination

        # Save result to cache
        self._cache[cache_key] = best_contamination
        return best_contamination

    def _evaluate_contamination(self, data, contamination, method):
        """Assess quality for contamination value"""
        try:
            if method == "isolation_forest":
                processed_data = self._apply_isolation_forest(data, contamination)
            elif method == "lof":
                processed_data = self._apply_lof(data, contamination)
            else:
                return contamination, float("inf")

            # Calculate quality metric
            residual_diff = data - processed_data
            score = np.mean(np.abs(residual_diff))
            return contamination, score
        except Exception:
            return contamination, float("inf")

    def _apply_isolation_forest(self, data, contamination):
        """Detect outliers using Isolation Forest"""
        if data is None or len(data) == 0 or len(data) < 5:
            return data

        # Prepare data
        values = data.values if hasattr(data, "values") else np.array(data)
        reshaped_data = values.reshape(-1, 1)

        # Optimal parameters depending on data size
        n_estimators, max_samples = get_optimal_isolation_forest_params(len(data),
                                                                        self.interval)  # Limit for large datasets

        # Create and train model with optimized parameters
        isol_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=13,
            n_jobs=-1,
        )

        # Determine outliers
        outliers = isol_forest.fit_predict(reshaped_data)
        outlier_mask = outliers == -1

        # Apply limit on outlier ratio
        outlier_mask = self._limit_outlier_ratio(data, outlier_mask)

        # Process outliers
        result = data.copy() if hasattr(data, "copy") else np.copy(data)

        if np.any(outlier_mask):
            # Replace outliers with median of non-outliers
            non_outlier_mask = ~outlier_mask
            if np.any(non_outlier_mask):
                median_value = np.median(values[non_outlier_mask])
            else:
                median_value = np.median(values)

            # Update values
            if hasattr(result, "iloc"):
                result.iloc[outlier_mask] = median_value
            else:
                result[outlier_mask] = median_value

        return result

    def _apply_lof(self, data, contamination, n_neighbors=None):
        """Detect outliers using Local Outlier Factor"""
        if data is None or len(data) == 0 or len(data) < 5:
            return data

        # Determine optimal number of neighbors
        data_len = len(data)
        if n_neighbors is None:
            n_neighbors = estimate_n_neighbors(data_len, self.interval)

        # Adjust for small datasets
        if n_neighbors >= data_len:
            n_neighbors = max(2, data_len // 2)

        if n_neighbors < 2:
            return data

        # Prepare data
        values = data.values if hasattr(data, "values") else np.array(data)
        reshaped_data = values.reshape(-1, 1)

        try:
            # Create and train model with optimized parameters
            lof = LocalOutlierFactor(
                contamination=contamination,
                n_neighbors=n_neighbors,
                metric="manhattan" if data_len < 50 else "minkowski",
                n_jobs=-1,
                algorithm="auto",  # Automatic algorithm selection
            )

            # Determine outliers
            outliers = lof.fit_predict(reshaped_data)
            outlier_mask = outliers == -1

            # Apply limit on outlier ratio
            outlier_mask = self._limit_outlier_ratio(data, outlier_mask)

            # Process outliers
            result = data.copy() if hasattr(data, "copy") else np.copy(data)

            if np.any(outlier_mask):
                # Replace outliers with median of non-outliers
                non_outlier_mask = ~outlier_mask
                if np.any(non_outlier_mask):
                    median_value = np.median(values[non_outlier_mask])
                else:
                    median_value = np.median(values)

                # Update values
                if hasattr(result, "iloc"):
                    result.iloc[outlier_mask] = median_value
                else:
                    result[outlier_mask] = median_value

            return result
        except Exception as e:
            logging.warning(f"Error when applying LOF: {e}")
            return data  # Return original data in case of error

    def _select_optimal_method(self, data, is_stationary):
        """Intelligent selection of optimal outlier detection method"""
        data_len = len(data)

        # Get adaptive thresholds based on interval
        small_threshold, medium_threshold, large_threshold = get_interval_thresholds(self.interval)

        # For small datasets LOF is better suited
        if data_len < small_threshold:
            return "lof"

        # For monthly data IsolationForest is better
        if self.interval in ["1M", "3M"]:
            return "isolation_forest"

        try:

            if is_stationary:
                clean_data = (
                    data.dropna()
                    if hasattr(data, "dropna")
                    else pd.Series(data[~np.isnan(data)])
                )

                # For stationary series with low variation - LOF
                if (
                        is_stationary
                        and clean_data.std() / abs(clean_data.mean() + 1e-10) < 0.5
                ):
                    return "lof"

            # By computational complexity IsolationForest is usually faster for large datasets
            if data_len > large_threshold:
                return "isolation_forest"

            # Analyze distribution shape - IsolationForest is better for skewed
            skew = abs(np.percentile(data, 75) - np.median(data)) / abs(
                np.median(data) - np.percentile(data, 25) + 1e-10
            )
            if skew > 1.5:  # Skewed distribution
                return "isolation_forest"

            # By default for medium datasets - LOF
            if data_len < medium_threshold:
                return "lof"

        except Exception:
            pass

        # By default for large datasets - IsolationForest
        return "isolation_forest"

    def decompose_and_remove_outliers(
            self,
            data: pd.Series,
            method="isolation_forest",
            contamination=None,
            n_neighbors=None,
            copy_data=True,
    ):
        """Decompose series and remove outliers from residual component"""
        if data is None or len(data) == 0:
            return data

        # Import decomposer and cache for reuse
        if not hasattr(self, "_trend_decomposer") or self._trend_decomposer is None:
            self._trend_decomposer = TrendDecomposer(self.interval)

        # Create working copy if needed
        working_data = data.copy() if copy_data and hasattr(data, "copy") else data

        try:
            # Decompose series using cached decomposer
            trend, seasonal, residual = self._trend_decomposer.decompose_series(
                working_data, method="STL"
            )

            self._metadata["is_combined"] = 1

            # Remove outliers from residual component
            if method == "isolation_forest":
                # Apply method to residual component
                residual_cleaned = self._process_with_method(
                    residual, "isolation_forest", contamination, is_residual=True
                )
            elif method == "lof":
                # Apply method to residual component
                residual_cleaned = self._process_with_method(
                    residual, "lof", contamination, n_neighbors, is_residual=True
                )
            else:
                raise ValueError("Unsupported method for outlier removal.")

            # Reconstruct cleaned series
            cleaned_series = trend + seasonal + residual_cleaned

            # Calculate outlier ratio (comparison with original series)
            self._calculate_outlier_ratio(data, cleaned_series)

            return cleaned_series

        except Exception as e:
            logging.error(f"Error during series decomposition: {e}")
            # In case of error return to direct method
            return self._process_with_method(
                data,
                "isolation_forest" if method == "isolation_forest" else "lof",
                contamination,
                n_neighbors,
            )

    def remove_outliers(
            self,
            data: pd.Series,
            method="auto",
            is_combined=None,
            contamination=None,
            n_neighbors=None,
            copy_data=True,
            is_stationary=False
    ):
        """
        Detect and remove outliers from time series

        Args:
            data: Time series
            method: Detection method ('isolation_forest', 'lof', 'combined', 'auto')
            is_combined: Combined method flag (True/False/None)
            contamination: Outlier ratio (if None, will be optimized)
            n_neighbors: Number of neighbors for LOF (if None, will be calculated)
            copy_data: Create data copy
            is_stationary: time series stationarity (expected to be determined in analyzer or from database)

        Returns:
            pd.Series: Cleaned time series
        """
        # Check input data
        if data is None or len(data) == 0:
            logging.warning("Received empty data for outlier detection")
            return data

        # Reset metadata
        self.reset_metadata()

        # Create working copy if needed
        working_data = data.copy() if copy_data and hasattr(data, "copy") else data

        # Determine combined approach mode
        if is_combined is None:
            use_direct_method = should_use_direct_method(len(data), self.interval)
            is_combined = not use_direct_method
            logging.debug(f"Auto-determined is_combined={is_combined} in detector")

        # Automatic method selection if method='auto'
        if method == "auto":
            method = self._select_optimal_method(working_data, is_stationary)
            logging.debug(f"Automatically selected outlier removal method: {method}")

        # Apply combined approach
        if is_combined:
            try:
                # Use specified method or select optimal
                preferred_method = (
                    method
                    if method in ["isolation_forest", "lof"]
                    else self._select_optimal_method(working_data, is_stationary)
                )

                # Perform decomposition and outlier removal
                return self.decompose_and_remove_outliers(
                    working_data,
                    method=preferred_method,
                    contamination=contamination,
                    n_neighbors=n_neighbors,
                    copy_data=False,
                )
            except Exception as e:
                # In case of decomposition error, use direct method
                logging.warning(
                    f"Error with combined method: {e}. Using direct method."
                )
                method = (
                    method
                    if method in ["isolation_forest", "lof"]
                    else self._select_optimal_method(working_data, is_stationary)
                )

        # Apply direct method for outlier processing
        self._metadata["is_combined"] = 0  # Explicit setting of direct method

        if method == "isolation_forest" or method == "combined":
            return self._process_with_method(
                working_data, "isolation_forest", contamination
            )
        elif method == "lof":
            return self._process_with_method(
                working_data, "lof", contamination, n_neighbors
            )
        else:
            # Default value for incorrect method
            logging.warning(f"Unknown method {method}, using isolation_forest")
            return self._process_with_method(
                working_data, "isolation_forest", contamination
            )