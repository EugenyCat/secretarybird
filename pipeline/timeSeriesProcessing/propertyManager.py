import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from pipeline.helpers.configs import PropertySourceConfig
from pipeline.helpers.db_funcs_orm import DBFuncsORM
from pipeline.helpers.utils import validate_required_locals
from pipeline.timeSeriesProcessing.preprocessingConfig import (
    TimeSeriesPreprocessingConfig,
)


class PropertyManager:
    """Time series property manager"""

    def __init__(self, interval, db_session):
        """
        Initialize property manager

        Args:
            interval: Time series interval
            db_session: Database session
        """
        validate_required_locals(["interval", "db_session"], locals())

        self.interval = interval
        self.db_session = db_session
        self.max_age_days = TimeSeriesPreprocessingConfig.get_max_age_for_interval(
            self.interval
        )
        self.source_tracking = {}
        logging.info(
            f"{self.__str__()} PropertyManager initialized: interval={interval}"
        )

    # === SIMPLIFIED METHODS FOR BASIC CASES ===

    def get_all_properties(self, ts_id: str, force_recalculate: dict = None) -> Dict:
        """
        Simplified method for getting all properties (without sources)

        Args:
            ts_id: Time series identifier
            force_recalculate: Dictionary of force recalculation flags

        Returns:
            Dict: Properties only (without sources)
        """
        force_recalculate = force_recalculate or {}
        properties, _ = self.get_properties(ts_id, force_recalculate)
        return properties

    def save_all_properties(self, ts_id: str, properties: Dict[str, Any]) -> bool:
        """
        Simplified method for saving properties

        Args:
            ts_id: Time series identifier
            properties: Dictionary of properties by groups

        Returns:
            bool: True if save successful
        """
        result = self.save_properties(ts_id, properties)
        return result is not None

    # === MAIN METHODS (REFACTORED) ===

    def get_properties(
        self, ts_id: str, force_recalculate: dict, version: Optional[str] = None
    ) -> Tuple[Dict, Dict]:
        """
        Get time series properties with source tracking
        """
        result = {}
        sources = {}

        # Determine groups to load
        groups_to_load = [
            group for group, force in force_recalculate.items() if not force
        ]
        if not groups_to_load:
            logging.info(
                f"{self.__str__()} All groups require recalculation for ts_id={ts_id}"
            )
            return result, sources

        logging.info(
            f"{self.__str__()} Getting properties for ts_id={ts_id}, groups={groups_to_load}"
        )

        try:
            # Get record from DB
            db_props = self._fetch_db_properties(ts_id, version)
            if not db_props:
                logging.info(
                    f"{self.__str__()} Property record for ts_id={ts_id} not found"
                )
                return result, sources

            # Process each group
            result, sources = self._process_property_groups(db_props, groups_to_load)

        except Exception as e:
            logging.error(
                f"{self.__str__()} Error getting properties: {str(e)}",
                exc_info=True,
            )

        # Update tracking
        self.source_tracking[ts_id] = sources
        logging.debug(f"{self.__str__()} Properties retrieved: {list(result.keys())}")
        return result, sources

    def save_properties(
        self,
        ts_id: str,
        properties: Dict[str, Any],
        version: Optional[str] = None,
        is_create_new_version: bool = False,
    ) -> Optional[str]:
        """
        Save time series properties to DB
        """
        if not self.db_session:
            logging.warning(
                f"{self.__str__()} No DB session available for saving properties."
            )
            return None

        try:
            property_id = version or str(uuid.uuid4())

            # Prepare data for saving
            props_to_save = self._prepare_save_data(ts_id, property_id, properties)

            # Create or update record
            success = self._create_or_update_record(
                ts_id, property_id, props_to_save, is_create_new_version
            )

            if success:
                self.db_session.commit()
                return property_id
            else:
                return None

        except Exception as e:
            self.db_session.rollback()
            logging.error(
                f"{self.__str__()} Failed to save properties: {e}", exc_info=True
            )
            return None

    # === HELPER METHODS ===

    def _fetch_db_properties(self, ts_id: str, version: Optional[str] = None):
        """Get property record from DB"""
        return DBFuncsORM.get_ts_preprocessing_property(
            self.db_session, {"ts_id": ts_id, "version": version}
        )

    def _process_property_groups(
        self, db_props, groups_to_load: List[str]
    ) -> Tuple[Dict, Dict]:
        """Process property groups from DB record"""
        result = {}
        sources = {}

        logging.debug(
            f"{self.__str__()} Record found: id={db_props.property_id}, calculated_at={db_props.calculated_at}"
        )

        for group_name in groups_to_load:
            # Extract group properties
            group_props = TimeSeriesPreprocessingConfig.extract_group_properties(
                db_props, group_name
            )

            if not group_props:
                logging.debug(
                    f"{self.__str__()} Properties for group {group_name} not found"
                )
                continue

            # Check if current
            if not self.is_stale_property(db_props, group_name):
                result[group_name] = group_props
                sources[group_name] = PropertySourceConfig.DATABASE
                logging.info(
                    f"{self.__str__()} Using current properties from DB for {group_name}"
                )
            else:
                logging.info(
                    f"{self.__str__()} Properties for {group_name} are stale - recalculation required"
                )

        return result, sources

    def _prepare_save_data(
        self, ts_id: str, property_id: str, properties: Dict[str, Any]
    ) -> Dict:
        """Prepare data for saving to DB"""
        now = datetime.now()

        # Basic metadata
        props_to_save = {
            "ts_id": ts_id,
            "property_id": property_id,
            "calculated_at": now,
            "is_active": 1,
        }

        # Add max_age fields
        for group_name, max_age in self.max_age_days.items():
            props_to_save[f"max_age_{group_name}"] = max_age

        # Process property groups
        for group_name, group_props in properties.items():
            if not group_props or not isinstance(group_props, dict):
                continue

            serialized_props = self._serialize_group_properties(group_name, group_props)
            props_to_save.update(serialized_props)

        return props_to_save

    def _serialize_group_properties(self, group_name: str, group_props: Dict) -> Dict:
        """Serialize group properties (JSON fields)"""
        result = {}

        group = TimeSeriesPreprocessingConfig.get_group_by_name(group_name)
        if not group:
            logging.warning(f"{self.__str__()} Unknown property group: {group_name}")
            return result

        for prop_name, prop_value in group_props.items():
            # Serialize JSON fields
            if (
                group.json_fields
                and prop_name in group.json_fields
                and prop_value is not None
            ):
                try:
                    prop_value = json.dumps(prop_value)
                except (TypeError, OverflowError) as e:
                    logging.warning(
                        f"{self.__str__()} Failed to serialize {prop_name}: {e}"
                    )
                    continue

            result[prop_name] = prop_value

        return result

    def _create_or_update_record(
        self,
        ts_id: str,
        property_id: str,
        props_to_save: Dict,
        is_create_new_version: bool,
    ) -> bool:
        """Create or update DB record"""
        # Check existing record
        existing_record = DBFuncsORM.find_ts_preprocessing_property_by_ids(
            self.db_session, {"ts_id": ts_id, "property_id": property_id}
        )

        if is_create_new_version or not existing_record:
            return self._create_new_version(ts_id, props_to_save, property_id)
        else:
            return self._update_existing_record(
                existing_record, props_to_save, property_id
            )

    def _create_new_version(
        self, ts_id: str, props_to_save: Dict, property_id: str
    ) -> bool:
        """Create new property version"""
        # Deactivate existing active versions
        count = DBFuncsORM.deactivate_active_ts_preprocessing_properties(
            self.db_session, {"ts_id": ts_id}
        )
        logging.info(f"{self.__str__()} Deactivated {count} properties for {ts_id}")

        # Create new record
        new_record = DBFuncsORM.create_ts_preprocessing_property(
            self.db_session, props_to_save
        )
        if new_record:
            logging.info(f"{self.__str__()} Created new property record: {property_id}")
            return True
        return False

    def _update_existing_record(
        self, existing_record, props_to_save: Dict, property_id: str
    ) -> bool:
        """Update existing record"""
        updated = DBFuncsORM.update_ts_preprocessing_property(
            self.db_session, {"record": existing_record, "data": props_to_save}
        )
        if updated:
            logging.info(
                f"{self.__str__()} Updated existing property record: {property_id}"
            )
            return True
        return False

    def is_stale_property(self, db_props, group_name):
        """
        Check if properties are stale for group
        """
        try:
            # Check force recalculation
            if self._check_force_recalculate(db_props, group_name):
                return True

            # Check by age
            return self._check_property_age(db_props, group_name)

        except Exception as e:
            logging.error(
                f"{self.__str__()} Error checking stale property for {group_name}: {e}"
            )
            return True  # In case of error, consider property stale

    def _check_force_recalculate(self, db_props, group_name: str) -> bool:
        """Check force recalculation flag"""
        force_recalc_field = f"force_recalc_{group_name}"
        force_recalc_value = getattr(db_props, force_recalc_field, None)

        if force_recalc_value is None:
            raise ValueError(
                f"Missing required field '{force_recalc_field}' in db_props"
            )

        # Convert to bool
        if isinstance(force_recalc_value, int):
            return force_recalc_value == 1
        elif isinstance(force_recalc_value, str):
            return force_recalc_value.lower() in ("true", "t", "yes", "y", "1")
        else:
            return bool(force_recalc_value)

    def _check_property_age(self, db_props, group_name: str) -> bool:
        """Check property age"""
        # Get max_age
        max_age_field = f"max_age_{group_name}"
        max_age = getattr(db_props, max_age_field, None)

        if max_age is None:
            raise ValueError(f"Missing required field '{max_age_field}' in db_props")
        if max_age <= 0:
            raise ValueError(f"Field '{max_age_field}' must be > 0, got: {max_age}")

        # Get and validate date
        calculated_at = self._parse_calculated_at(db_props)

        # Check staleness
        days_old = max(0, (datetime.now() - calculated_at).days)
        return days_old > max_age

    def _parse_calculated_at(self, db_props) -> datetime:
        """Parse calculated_at field"""
        calculated_at = getattr(db_props, "calculated_at", None)

        if calculated_at is None:
            raise ValueError("Missing required field 'calculated_at' in db_props")

        if not isinstance(calculated_at, datetime):
            if isinstance(calculated_at, str):
                try:
                    calculated_at = datetime.fromisoformat(
                        calculated_at.replace(" ", "T")
                    )
                except Exception as e:
                    raise ValueError(f"Invalid datetime format in 'calculated_at': {e}")
            else:
                raise ValueError(
                    f"Field 'calculated_at' must be datetime or string, got: {type(calculated_at)}"
                )

        return calculated_at

    def __str__(self):
        return "[timeSeriesProcessing/propertyManager.py]"