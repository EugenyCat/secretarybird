import logging
from etl_pipeline.etl_manager.baseManager import BaseManager
from etl_pipeline.helpers.utils import ETLUtils
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import math


class DataQualityManager(BaseManager):
    """
        DataQualityManager is responsible for managing data quality metrics
        within a specified data source. It provides functionalities to check
        for missing data, data continuity, and duplicates, as well as to recover
        any missing values and generate comprehensive reports on data quality.

        Attributes:
            __unit_to_seconds (dict): A mapping of time intervals to their equivalent
                                       in seconds, used for interval calculations.
    """

    # Mapping of time intervals to their equivalent in seconds
    __unit_to_seconds = {
        "1h": 3600,        # 1 hour in seconds
        "12h": 12 * 3600,  # 12 hours in seconds
        "1d": 24 * 3600,   # 1 day in seconds
        "3d": 72 * 3600,   # 3 days in seconds
        "1w": 7 * 24 * 3600  # 1 week in seconds
    }

    def __init__(self, source_name):
        """
            Initializes a DataQualityManager instance for a specified data source.

            This method sets up the manager for monitoring and managing data quality
            for the provided source name by calling the parent class initializer.

            Args:
                source_name (str): The name of the data source for which the quality manager is being initialized.
        """
        super().__init__(source_name)


    def get_the_known_lacks(self):
        """
            Read the file that contains the existed lacks
        """
        file_path = f"{os.getenv('LACKS_IN_DB_PATH')}/{self.currency}_lacks.csv"

        if os.path.exists(file_path):
            known_lacks = pd.read_csv(file_path)
            known_lacks['month_where_lacks'] = pd.to_datetime(known_lacks['month_where_lacks'])
            return known_lacks

        return pd.DataFrame(columns=['month_where_lacks', 'interval', 'currency', 'rows_amount', 'data_created'])


    def check_if_db_has_lacks(self):
        """
            Checking if there are some lacks in CH db
        """

        ch_response_df = self.db_funcs.get_lacks()

        if len(ch_response_df) > 0:
            ch_response_df['month_where_lacks'] = pd.to_datetime(ch_response_df['month_where_lacks'])

            known_lacks = self.get_the_known_lacks()

            new_lacks = ch_response_df.merge(known_lacks, on=['month_where_lacks', 'interval', 'currency', 'rows_amount'], how='left', indicator=True)
            new_lacks = new_lacks[new_lacks['_merge'] == 'left_only'].drop(columns='_merge')

            new_lacks['data_created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            merged_lacks = pd.concat([known_lacks, new_lacks], axis=0).sort_values(by='month_where_lacks')

            merged_lacks.to_csv(f"{os.getenv('LACKS_IN_DB_PATH')}/{self.currency}_lacks.csv", index=False)
            return new_lacks

        return []


    def check_if_db_continuous(self):
        """
            Checking is there a CH db has all months without missing
        """

        # Define min data from db
        start_time = self.db_funcs.get_extreme_date(method='min')

        # Define last data from db
        end_time = self.db_funcs.get_extreme_date()

        # Calculate the amount of rows for the currency and interval
        db_amount_of_time_units = self.db_funcs.get_amount_of_rows_for_currency_and_interval()

        # Calculate correct difference between two moments of time
        if self.interval == '1M':
            correct_time_unit_amount = ETLUtils.count_months(start_time, end_time)
        else:
            correct_time_unit_amount = (
                    (end_time - start_time).total_seconds() / self.__unit_to_seconds[f'{self.interval}'] + 1
            )
        difference = math.fabs(db_amount_of_time_units - correct_time_unit_amount)

        return db_amount_of_time_units, correct_time_unit_amount, difference


    def get_data_from_db(self):
        """
            Send a query to CH and get all data over currency, interval
        """
        # Get data from db
        df_currency = self.db_funcs.get_data()

        df_currency['Open_time'] = pd.to_datetime(df_currency['Open_time'])
        df_currency.set_index('Open_time', inplace=True)

        return df_currency


    def get_data_range(self):
        """
            Return the whole data range from `min_time` to `max_time` with `interval`
        """
        # Define the max and min data from db
        max_time = self.db_funcs.get_extreme_date()
        min_time = self.db_funcs.get_extreme_date(method='min')

        if self.interval == '1M':
            freq = '1MS'
        elif self.interval == '1w':
            freq = '1W-MON'
        else:
            freq = self.interval

        # Create the whole data range
        return pd.DataFrame(pd.date_range(start=min_time, end=max_time, freq=freq),
                                 columns=['Open_time']).set_index('Open_time')


    def calculate_close_time(self, row):
        """
            Calculate the close time based on the interval
        """
        open_time = row['Open_time']
        interval = row['Interval']

        if interval == '1h':
            delta = timedelta(hours=1)
        elif interval == '12h':
            delta = timedelta(hours=12)
        elif interval == '1d':
            delta = timedelta(days=1)
        elif interval == '3d':
            delta = timedelta(days=3)
        elif interval == '1w':
            delta = timedelta(weeks=1)
        elif interval == '1M':
            delta = relativedelta(months=1)
        else:
            raise ValueError(f"Unknown interval: {interval}")

        return open_time + delta - timedelta(seconds=1)


    def fill_missing_data(self):
        """
            Fill the values that are missing in time series
        """
        # Get data from db
        df_currency = self.get_data_from_db()

        # Create the whole data range
        full_time = self.get_data_range()

        # Merge the currency data and whole data range
        numeric_columns = [
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Quote_asset_volume',
            'Number_of_trades',
            'Taker_buy_base_asset_volume',
            'Taker_buy_quote_asset_volume'
        ]
        df_currency_with_whole_datas = full_time.merge(df_currency, on='Open_time', how='outer')[numeric_columns]

        # Convert type of columns from object into float
        for col in df_currency_with_whole_datas.select_dtypes(include='object').columns:
            df_currency_with_whole_datas[col] = pd.to_numeric(df_currency_with_whole_datas[col], errors='coerce')

        # Set DatetimeIndex
        df_currency_with_whole_datas.index = pd.DatetimeIndex(df_currency_with_whole_datas.index)

        # Fill missing data using pandas interpolate (method='time')
        df_currency_with_whole_datas.interpolate(method='time', inplace=True)

        # Set columns `Currency` and `Interval`
        df_currency_with_whole_datas['Currency'] = self.currency
        df_currency_with_whole_datas['Interval'] = self.interval

        # Set columns `Close_time`
        df_currency_with_whole_datas = df_currency_with_whole_datas.merge(df_currency[['Close_time', 'Source']], on='Open_time', how='left')

        # Calculate the `Close_time` where a value is missing
        df_currency_with_whole_datas['Open_time'] = df_currency_with_whole_datas.index
        df_currency_with_whole_datas['Close_time'] = df_currency_with_whole_datas.apply(
            lambda row: self.calculate_close_time(row) if pd.isna(row['Close_time']) else row['Close_time'],
            axis=1
        )

        df_currency_with_whole_datas['Number_of_trades'] = df_currency_with_whole_datas['Number_of_trades'].astype(int)

        # Calculate the `Close_time` where a value is missing
        df_currency_with_whole_datas['is_return'] = df_currency_with_whole_datas['Source'].fillna(1)
        df_currency_with_whole_datas['Source'] = df_currency_with_whole_datas['Source'].fillna('pandas_interpolate')

        return df_currency_with_whole_datas[df_currency_with_whole_datas['is_return'] == 1].drop(columns=['is_return'])


    def insert_into_db(self, df_currency):
        """
            Insert data into ch
        """
        column_names = [column[0] for column in self.db_funcs.table_columns]
        data_tuples = [list(x) for x in df_currency[column_names].to_numpy()]
        self.db_funcs.insert_data(data_tuples)


    def generate_report_for_checking_if_latest_date_is_not_today(self):
        """
            Generates a report on whether the latest date in the database is not the current date.

            Checks if the latest recorded date in the database matches the current date based on the specified interval ('1h' or '1d').

            :return:
                - answer_report (str): A message report indicating any discrepancies in dates.
                - is_data_relevant (int): A flag indicating whether the data is relevant (1 - relevant, 0 - not relevant).
        """
        # Initialize answer_report message
        answer_report = ''

        # Define the last data in the database
        latest_db_date = self.db_funcs.get_extreme_date().strftime('%Y-%m-%d %H:%M:%S')

        # Define current time in UTC
        current_utc_time = None
        if self.interval == '1h':
            current_utc_time = datetime.now(timezone.utc).replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
        elif self.interval == '1d':
            current_utc_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).strftime(
                '%Y-%m-%d %H:%M:%S')

        # Flag indicating if latest_db_date equals today
        is_data_relevant = True

        # Generate a report
        if current_utc_time != latest_db_date and self.interval in ['1h', '1d']:
            is_data_relevant = False
            answer_report += f" ➜ 🚨🚨 <b>Current data doesn't equal the latest db value</b> `Open_time`. \n"
            answer_report += f"Today: {current_utc_time} || Last `Open_time`: {latest_db_date}\n"

        return answer_report, is_data_relevant


    def generate_report_for_checking_if_db_continuous(self):
        """
            Generates a report on whether the database contains continuous data for all required months.

            :return: A tuple containing:
                - answer_report (str): A message indicating the continuity status.
                - is_data_continuous (int): A flag (1 if continuous, 0 if not).
        """
        # Initialize answer_report message
        answer_report = ''

        # Define if the CH db has all months without missing
        db_continuous = self.check_if_db_continuous()

        # Flag indicating if db continuous
        is_data_continuous = True

        # Generate a report
        if db_continuous[2] > 0:
            is_data_continuous = False
            answer_report += f" ➜ There is the <b>discontinuity</b> of data at  {db_continuous[2]} units.\n"

        return answer_report, is_data_continuous


    def generate_report_for_checking_if_db_has_lacks(self):
        """
            Generates a report on whether the database has any missing data entries.

            :return: A tuple containing:
                - answer_report (str): A message detailing any data lacks found.
                - is_db_without_lacks (int): A flag (1 if no lacks, 0 if lacks are present).
        """
        # Initialize answer_report message
        answer_report = ''

        # Define if the CH db has some lacks in data
        db_lacks = self.check_if_db_has_lacks()

        # Flag indicating if CH db has some lacks
        is_db_without_lacks = True

        # Generate a report
        if len(db_lacks) > 0:
            is_db_without_lacks = False
            answer_report += f" ➜ The next <b>lacks</b> are found: \n{db_lacks.to_string(index=False)}\n"

        return answer_report, is_db_without_lacks


    def recover_and_report_missing_data(self):
        """
        Fills in the missing data in the time series and generates a report
        detailing the recovered data.

        :return: str: A report summarizing the missing data that was recovered.
        """
        # Retrieve the missing values in the time series
        missing_data = self.fill_missing_data()

        # Insert the recovered missing data into the database table
        self.insert_into_db(missing_data)

        # Generate the report of the recovered missing data
        answer_report = self.line_str
        answer_report += (f"✅ Missing data recovered: "
                          f"{[el.strftime('%Y-%m-%d %H:%M:%S') for el in missing_data['Open_time'][-5:]]}. \n")

        return answer_report


    def report_and_remove_duplicates_in_db(self):
        """
            Checks for duplicate records in the database, generates a report on the findings,
            and removes the duplicates if found.

            :return: str: A report summarizing the presence of duplicates and the result of their removal.
        """
        # Initialize the report message
        answer_report = ''

        # Check for duplicate entries in the database tables
        df_doubles = self.db_funcs.define_doubles_in_tables()

        # Generate the report if duplicates are found
        if len(df_doubles) > 0:
            answer_report += self.line_str
            answer_report += (f" ➜ Duplicates found:\n"
                              f"{[el.strftime('%Y-%m-%d %H:%M:%S') for el in df_doubles['Open_time'][-5:]]}. \n")

            # Remove duplicate entries from the database
            self.db_funcs.remove_duplicates()

            # Verify if duplicates have been successfully removed
            if len(self.db_funcs.define_doubles_in_tables()) == 0:
                answer_report += "✅ Duplicates have been successfully deleted. \n"
            else:
                answer_report += "🚨🚨 Duplicates <b>were NOT</b> deleted. \n"

        return answer_report


    def generate_data_quality_report(self):
        """
            Generates a comprehensive report about potential data quality issues in the database,
            including missing dates, data discontinuities, and duplicates.

            The function checks:
            1. Whether the latest date in the database is today's date.
            2. If there are any gaps or discontinuities in the data. (e.g. all months, days etc.)
            3. Whether there are any missing or lacking records in the database. (1 row: 2024-01-01 09:00:00, 2 row 2024-01-01 12:00:00 - miss 10:00, 11:00)
            4. If duplicates are present and, if so, removes them.

            If missing data is identified, it attempts to recover it and updates the report accordingly.

            :return: str: A detailed data quality report about potential issues in the database, or an empty string if no issues are found.
        """

        # Generate report about missing or incorrect latest date
        report_if_latest_date_is_not_today, is_data_relevant = self.generate_report_for_checking_if_latest_date_is_not_today()

        # Generate report about data continuity
        report_if_db_continuous, is_data_continuous = self.generate_report_for_checking_if_db_continuous()

        # Generate report about missing data in the database
        report_if_db_has_lacks, is_db_without_lacks = self.generate_report_for_checking_if_db_has_lacks()

        # Combine all reports into one
        answer_report = report_if_latest_date_is_not_today + report_if_db_continuous + report_if_db_has_lacks

        # If data is not continuous or the database has lacks, attempt to fill missing data and report
        if not is_data_continuous or not is_db_without_lacks:
            answer_report += self.recover_and_report_missing_data()

        # Check for and remove duplicates in the database, updating the report accordingly
        answer_report += self.report_and_remove_duplicates_in_db()

        # If any issues were reported, return the full report, otherwise return an empty string
        if len(answer_report) > 0:
            # Initialize the report title with currency and interval information
            report_title = f'📢🔔⚠️ REPORT about problems for <b>{self.currency}, {self.interval}</b>:\n'
            return report_title + answer_report + self.line_split_messages

        return None


class DataQualityManagerFacade(DataQualityManager):
    """
        Facade for managing data quality checks on a database.

        This class encapsulates the functionality of performing data quality checks
        for a specified currency and time interval, and provides a simplified interface
        for interacting with the underlying data quality management system.
    """

    def __init__(self, source_name):
        """
            Initializes the DataQualityManagerFacade with a given source name.

            :param source_name: str - The name of the data source being managed.
        """
        super().__init__(source_name)


    def check_db(self, currency, interval):
        """
            This function performs a data quality check on the database for a given currency and time interval.
            It establishes a database session, configures the currency and interval, and uses the DBFuncs class
            to perform various operations such as checking for data quality issues (missing data, duplicates, etc.).

            If any data quality issues are found, a report is generated, and the function returns an error message.
            If no issues are found, it returns a success message.

            :param currency: str - The currency for which the database should be checked.
            :param interval: str - The time interval (e.g., '1h', '1d') for which the database should be checked.

            :return:
                - (None, dict) if there is an error or data quality issue.
                - (dict, None) if no issues are found in the database.
        """
        try:
            # Set the currency and interval for the DB check
            self.set_currency(currency) \
                .set_interval(interval)

            # Set the currency and interval information in db_funcs
            self.db_funcs.set_currency(self.currency) \
                .set_interval(self.interval)

            # Generate a data quality report, checking for issues like missing data, duplicates, etc.
            data_quality_report = self.generate_data_quality_report()

            # If a data quality report is generated (indicating issues), return an error status and message
            if data_quality_report:
                self.notifier.notify(data_quality_report)
                message = f'Data quality issues detected for {currency}, {interval}'
                logging.info(f'{message}')
                return None, {'status': 'error', 'message': message}

            # If no issues are found, return a success message
            return {'status': 'success', 'message': f'{self.__str__()}: Check db process didn\'t find any issues.'}, None

        except Exception as e:
            # Log and return an error message in case of an exception
            message = f'{self.__str__()}: The data checking and missing data filling process finished with an error: {e}'
            self.notifier.notify(message)
            logging.info(f'{message}: {e}')
            return None, {'status': 'error', 'message': message}

        finally:
            # Close the database session and reset db_funcs after the check is complete
            self.clickhouse_conn.close_session()
            self.db_funcs = None


    def __str__(self):
        return f'[etl_manager/dataQualityManager.py] {self.currency}_{self.interval}'


"""
obj = DataQualityManager('binance_api', 'CRYPTO')
ans = obj.check_db('adausdt', '1M')
print(ans)
"""