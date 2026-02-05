import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

from pipeline.timeSeriesProcessing.baseModule.baseProcessor import BaseProcessor


class ProcessingPipeline:
    """Time series processing pipeline"""

    def __init__(self, processors: List[BaseProcessor] = None):
        """Initialize pipeline with processors"""
        self.processors = processors or []

    def addProcessor(self, processor: BaseProcessor) -> "ProcessingPipeline":
        """Add processor to pipeline"""
        self.processors.append(processor)
        return self

    def process(
        self, data: pd.DataFrame, context: Dict[str, Any], trace_helper=None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Process data through all pipeline stages"""
        currentData = data.copy()

        # Mapping processors to names for tracing
        processor_trace_names = {
            "AnalysisProcessor": "01_after_analyzer",
            "PeriodicityDetectorProcessor": "02_after_periodicity",
            "DecompositionProcessor": "03_after_decomposition",
            "OutlierDetectionProcessor": "04_after_outlier",
        }

        for i, processor in enumerate(self.processors):
            try:
                processor_name = processor.__class__.__name__
                logging.info(f"Stage {i + 1}: {processor_name}")
                currentData, context = processor.process(currentData, context)
                logging.info(f"Stage {i + 1}: {processor_name} - completed")

                # 🔍 TRACE: After each processor
                if trace_helper and processor_name in processor_trace_names:
                    try:
                        # Save DataFrame
                        trace_helper.save_df(
                            currentData, processor_trace_names[processor_name]
                        )

                        # Save context (only properties, without pandas objects)
                        context_snapshot = {
                            "processor": processor_name,
                            "propertySources": context.get("propertySources", {}),
                            "currentProperties": self._make_json_safe(
                                context.get("currentProperties", {})
                            ),
                        }
                        context_name = processor_trace_names[processor_name].replace(
                            "after_", "context_"
                        )
                        trace_helper.save_context(context_snapshot, context_name)

                    except Exception as e:
                        logging.warning(f"Failed to trace {processor_name}: {e}")

                # Check for critical errors after each processor
                if context.get("error"):
                    error_info = context["error"]
                    logging.error(
                        f"Critical error at stage {error_info['stage']}: {error_info['message']}"
                    )
                    break

            except Exception as e:
                logging.error(
                    f"Error in processor {processor.__class__.__name__}: {e}"
                )
                raise

        return currentData, context

    def _make_json_safe(self, obj):
        """
        Recursively convert object to JSON-compatible format.

        Removes pandas Series/DataFrame and other non-serializable objects.
        """
        from enum import Enum

        import numpy as np
        import pandas as pd

        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return f"<{type(obj).__name__} shape={obj.shape}>"
        elif isinstance(obj, np.ndarray):
            return f"<ndarray shape={obj.shape}>"
        elif isinstance(obj, dict):
            return {str(k): self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_safe(item) for item in obj]
        else:
            # For other objects - string representation
            return str(obj)