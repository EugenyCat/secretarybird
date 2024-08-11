import logging

from pipeline import ExtractBase
from database.ClickHouseFuncs import (
    get_extreme_date,
    get_lacks,
    get_amount_of_rows_for_currency_and_interval,
    define_doubles_in_tables,
    remove_doubles,
    get_data,
    insert_data,
    column_names
)
import pandas as pd
import os
import json
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
import pandas
import math


class ETLManager(ExtractBase):
    def __init__(self, source_name, database):
        super().__init__()
        # database
        self.database = database
        # init data that contains the info about api (currencies, intervals etc)
        self.api_configurations = json.load(open(f"{os.getenv('JSON_EXTRACT_CURRENCIES_SETTINGS')}"))[source_name]
        # info for defining the lacks in CH db
        self.valid_amount_of_rows = {
            "1h": 24,
            "12h": 2,
            "1d": 1,
            "3d": [9, 10],
            "1w": 4,
            "1M": 1
        }
        # how many seconds in interval
        self.unit_to_seconds = {
            "1h": 3600,
            "12h": 12 * 3600,
            "1d": 24 * 3600,
            "3d": 72 * 3600,
            "1w": 7 * 24 * 3600
        }



    def get_api_configurations(self):
        return self.api_configurations


    """
    def get_where_condition(self, interval):
        ""
            Generate WHERE condition depends on `interval`
        ""
        row_num_coef = self.valid_amount_of_rows[f'{interval}']

        if interval in ['1h', '12h', '1d']:
            where_condition = f""
                            WHERE   (rows_amount < {row_num_coef * 31} AND month_name in [1, 3, 5, 7, 8, 10, 12]) 
                                    OR 
                                    (rows_amount < {row_num_coef * 30} AND month_name in [4, 6, 9, 11])
                                    OR 
                                    (rows_amount < {row_num_coef * 29} AND month_name = 2 AND is_leap = 1)
                                    OR 
                                    (rows_amount < {row_num_coef * 28} AND month_name = 2 AND is_leap = 0)
                    ""
        elif interval == '3d':
            where_condition = f""
                            WHERE   (rows_amount < {row_num_coef[0]} AND month_name = 2) 
                                    OR 
                                    (rows_amount < {row_num_coef[1]} AND month_name != 2)
                    ""
        elif interval in ['1w', '1M']:
            where_condition = f""
                            WHERE   rows_amount < {row_num_coef}
                    ""
        else:
            raise Exception("SOMETHING WRONG! BROKE THE REQUEST!")

        return where_condition


    def get_lacks(self, currency, interval):
        where_condition = self.get_where_condition(interval)

        # TODO: hide this request
        get_records_count_request = f""
                        SELECT month_where_lacks,
                                '{interval}' as "interval",
                                '{currency}' as "currency",
                                rows_amount
                        FROM (
                            SELECT 
                                toStartOfMonth(Open_time) AS month_where_lacks,
                                MONTH(Open_time) AS month_name,
                                IF (MONTH(Open_time) = 2 AND DAY(toLastDayOfMonth(Open_time)) = 28, 0,
                                    IF (MONTH(Open_time) = 2 AND DAY(toLastDayOfMonth(Open_time)) = 29, 1,
                                        NULL 
                                )) AS is_leap,
                                count(*) AS rows_amount
                            FROM {self.database}{interval}.{currency.lower()}
                            WHERE Open_time < toStartOfMonth(today())
                                    and Open_time >= (
                                        SELECT toStartOfMonth(addMonths(MIN(Open_time), 1))
                                        FROM {self.database}{interval}.{currency.lower()}
                                    )
                            GROUP BY month_where_lacks, month_name, is_leap
                            ORDER BY rows_amount
                        )
                        {where_condition}
                ""

        ch_response_df = pandas.DataFrame(
            self.db_session.query(get_records_count_request).result_rows,
            columns=['month_where_lacks', 'interval', 'currency', 'rows_amount']
        )

        return ch_response_df
        """

    def get_the_known_lacks(self, currency):
        """
            Read the file that contains the existed lacks
        """
        file_path = f"{os.getenv('LACKS_IN_DB_PATH')}/{currency}_lacks.csv"

        if os.path.exists(file_path):
            known_lacks = pd.read_csv(file_path)
            known_lacks['month_where_lacks'] = pd.to_datetime(known_lacks['month_where_lacks'])
            return known_lacks

        return pd.DataFrame(columns=['month_where_lacks', 'interval', 'currency', 'rows_amount', 'data_created'])


    def check_if_db_has_lacks(self, currency, interval):
        """
            Checking if there are some lacks in CH db
        """

        ch_response_df = get_lacks(self.db_session, self.database, currency, interval)

        if len(ch_response_df) > 0:
            ch_response_df['month_where_lacks'] = pd.to_datetime(ch_response_df['month_where_lacks'])

            known_lacks = self.get_the_known_lacks(currency)

            new_lacks = ch_response_df.merge(known_lacks, on=['month_where_lacks', 'interval', 'currency', 'rows_amount'], how='left', indicator=True)
            new_lacks = new_lacks[new_lacks['_merge'] == 'left_only'].drop(columns='_merge')

            #new_lacks['is_new_lacks'] = 1
            new_lacks['data_created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            merged_lacks = pd.concat([known_lacks, new_lacks], axis=0).sort_values(by='month_where_lacks')

            merged_lacks.to_csv(f"{os.getenv('LACKS_IN_DB_PATH')}/{currency}_lacks.csv", index=False)
            return new_lacks

        return []


    def check_if_db_continuous(self, currency, interval):
        """
            Checking is there a CH db has all months without missing
        """

        # Define min data from db
        start_time = get_extreme_date(self.db_session, self.database, currency, interval, method='min')

        # Define last data from db
        end_time = get_extreme_date(self.db_session, self.database, currency, interval)

        # Calculate the amount of rows for the currency and interval
        db_amount_of_time_units = get_amount_of_rows_for_currency_and_interval(self.db_session, self.database, currency, interval)
        """
        db_amount_of_time_units = self.db_session.query(f""
                    SELECT count(Open_time) 
                    FROM {self.database}{interval}.{currency.lower()}
                "").result_rows[0][0]
        """


        # Calculate correct difference between two moments of time
        if interval == '1M':
            correct_time_unit_amount = self.count_months(start_time, end_time)
        else:
            correct_time_unit_amount = (end_time - start_time).total_seconds() / self.unit_to_seconds[f'{interval}'] + 1
        difference = math.fabs(db_amount_of_time_units - correct_time_unit_amount)

        return db_amount_of_time_units, correct_time_unit_amount, difference


    """
    def get_extreme_date(self, currency, interval, method='max'):
        end_time = self.db_session.query(f""
                    SELECT {method}(Open_time) 
                    FROM {self.database}{interval}.{currency.lower()}
                "").result_rows[0][0]

        return end_time
    """

    """
    def define_doubles_in_tables(self, currency, interval):
        ""
            Send a query to CH and get all doubles data over currency, interval
        ""
        # Get data from db
        ch_response = self.db_session.query(f""
                                    SELECT Open_time, COUNT(*) AS count
                                    FROM {self.database}{interval}.{currency.lower()}
                                    GROUP BY Open_time
                                    HAVING count > 1
                                "")

        # Create a df
        df_currency = pd.DataFrame(ch_response.result_rows, columns=ch_response.column_names)

        return df_currency
    """

    """
    def remove_doubles(self, currency, interval):
        ""
            Rewrite the table with uniq values
        ""
        # Create table
        create_tab_request = f"CREATE TABLE {self.database}{interval}.{currency.lower()}_no_duplicates  ( \
        	                                                Currency String, \
        	                                                Interval String, \
                                                            Open_time DateTime, \
                                                            Open Float64, \
                                                            High Float64, \
                                                            Low Float64, \
                                                            Close Float64, \
                                                            Volume Float64, \
                                                            Close_time DateTime, \
                                                            Quote_asset_volume Float64, \
                                                            Number_of_trades UInt32, \
                                                            Taker_buy_base_asset_volume Float64, \
                                                            Taker_buy_quote_asset_volume Float64, \
                                                            Source String DEFAULT 'binance_api'\
                                            ) ENGINE = MergeTree() \
                                            PARTITION BY toYYYYMM(Open_time) \
                                            ORDER BY (Open_time);"

        self.db_session.query(create_tab_request)

        # Insert table
        insert_tab_request = f""
            INSERT INTO {self.database}{interval}.{currency.lower()}_no_duplicates
            SELECT
            any(Currency) as Currency,
            any(Interval) as Interval,
            Open_time,
            any(Open) as Open,
            any(High) as High,
            any(Low) as Low,
            any(Close) as Close,
            any(Volume) as Volume,
            any(Close_time) as Close_time,
            any(Quote_asset_volume) as Quote_asset_volume,
            any(Number_of_trades) as Number_of_trades,
            any(Taker_buy_base_asset_volume) as Taker_buy_base_asset_volume,
            any(Taker_buy_quote_asset_volume) as Taker_buy_quote_asset_volume,
            any(Source) as Source
            FROM {self.database}{interval}.{currency.lower()}
            GROUP BY 
            Open_time
            ""

        self.db_session.query(insert_tab_request)

        # Rename table
        rename_request = f""
            RENAME TABLE {self.database}{interval}.{currency.lower()} TO {self.database}{interval}.{currency.lower()}_old, {self.database}{interval}.{currency.lower()}_no_duplicates TO {self.database}{interval}.{currency.lower()};
        ""
        self.db_session.query(rename_request)

        # Drop table
        drop_request = f""
                    DROP TABLE {self.database}{interval}.{currency.lower()}_old;
                ""
        self.db_session.query(drop_request)
    """

    def check_db(self, currency, interval):
        """
            Check CH db data on:
            - if data has lacks like (1 row: 2024-01-01 09:00:00, 2 row 2024-01-01 12:00:00 - miss 10:00, 11:00)
            - if data is continuous (e.g. all months, days etc. - similar prev. case)
            - if  last data from dt is really today
        """
        try:
            self.db_session = self.ch_connection_obj.get_session()

            db_lacks = self.check_if_db_has_lacks(currency, interval)
            db_continuous = self.check_if_db_continuous(currency, interval)
            last_db_data = get_extreme_date(self.db_session, self.database, currency, interval).strftime('%Y-%m-%d %H:%M:%S')

            # Define current time in UTC
            current_utc_time = None
            if interval == '1h':
                current_utc_time = datetime.now(timezone.utc).replace(minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
            elif interval == '1d':
                current_utc_time = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).strftime('%Y-%m-%d %H:%M:%S')
            is_data_relevant = 1
            is_data_continuous = 1
            is_db_without_lacks = 1

            line_split_messages = '\n✦•·················•✦•·················•✦\n'
            line_str = '⤵\n'
            title = f'📢🔔⚠️REPORT about problems for <b>{currency}, {interval}</b>:\n'
            answer_report = ''

            if current_utc_time != last_db_data and interval in ['1h', '1d']:
                is_data_relevant = 0
                answer_report += f" ➜ 🚨🚨 <b>Current data doesn't equal the last db value</b> `Open_time`. \n"
                answer_report += f"Today: {current_utc_time} || Last `Open_time`: {last_db_data}\n"
            if db_continuous[2] > 0:
                is_data_continuous = 0
                answer_report += f" ➜ There is the <b>discontinuity</b> of data at  {db_continuous[2]} units.\n"
            if len(db_lacks) > 0:
                is_db_without_lacks = 0
                answer_report += f" ➜ The next <b>lacks</b> are found: \n{db_lacks.to_string(index=False)}\n"
            # Call fill_missing_data() if is_data_continuous = 1 or is_db_without_lacks = 1
            # if data isn't relevant the method isn't called
            if not is_data_continuous or not is_db_without_lacks:  # or not is_not_data_relevant
                missing_data = self.fill_missing_data(currency, interval)
                self.insert_into_db(missing_data, currency, interval)
                answer_report += line_str
                answer_report += (f"✅ Missing data recovered:\n{[el.strftime('%Y-%m-%d %H:%M:%S') for el in missing_data['Open_time'][-5:]]}. \n")

            # check if there are doubles in data
            df_doubles = define_doubles_in_tables(self.db_session, self.database, currency, interval)
            """
            df_doubles = self.define_doubles_in_tables(currency, interval)
            """
            if len(df_doubles) > 0:
                answer_report += line_str
                answer_report += f" ➜ Doubles are found:\n{[el.strftime('%Y-%m-%d %H:%M:%S') for el in df_doubles['Open_time'][-5:]]}. \n"
                remove_doubles(self.db_session, self.database, currency, interval)
                """
                    self.remove_doubles(currency, interval)
                """
                if len(define_doubles_in_tables(self.db_session, self.database, currency, interval)) == 0:
                    answer_report += f"✅ Doubles are deleted. \n"
                else:
                    answer_report += f"🚨🚨 Doubles <b>AREN'T</b> deleted. \n"
            if len(answer_report) > 0:
                return None, {'status': 'error', 'message': title + answer_report + line_split_messages}
            return {'status': 'success', 'message': 'Check db process didn\'t find the lacks.'}, None
        except Exception as e:
            message = f'🚨🚨 The checking data and the filling missing processes for <b>{currency}, {interval}</b> <b>finished with mistake {e}</b>'
            logging.info(f'{message}: {e}')
            return None, {'status': 'error', 'message': message}
        finally:
            self.ch_connection_obj.close_session()


    def get_data_from_db(self, currency, interval):
        """
            Send a query to CH and get all data over currency, interval
        """
        # Get data from db
        df_currency = get_data(self.db_session, self.database, currency, interval)

        df_currency['Open_time'] = pd.to_datetime(df_currency['Open_time'])
        df_currency.set_index('Open_time', inplace=True)

        return df_currency


    def get_data_range(self, currency, interval):
        """
            Return the whole data range from `min_time` to `max_time` with `interval`
        """
        # Define the max and min data from db
        max_time = get_extreme_date(self.db_session, self.database, currency, interval)
        min_time = get_extreme_date(self.db_session, self.database, currency, interval, method='min')

        if interval == '1M':
            freq = '1MS'
        elif interval == '1w':
            freq = '1W-MON'
        else:
            freq = interval

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
            close_time = open_time + timedelta(hours=1) - timedelta(seconds=1)
        elif interval == '12h':
            close_time = open_time + timedelta(hours=12) - timedelta(seconds=1)
        elif interval == '1d':
            close_time = open_time + timedelta(days=1) - timedelta(seconds=1)
        elif interval == '3d':
            close_time = open_time + timedelta(days=3) - timedelta(seconds=1)
        elif interval == '1w':
            close_time = open_time + timedelta(weeks=1) - timedelta(seconds=1)
        elif interval == '1M':
            close_time = open_time + relativedelta(months=1) - timedelta(seconds=1)
        else:
            raise ValueError(f"Unknown interval: {interval}")

        return close_time


    def fill_missing_data(self, currency, interval):
        """
            Fill the values that are missing in time series
        """
        # Get data from db
        df_currency = self.get_data_from_db(currency, interval)

        # Create the whole data range
        full_time = self.get_data_range(currency, interval)

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
        df_currency_with_whole_datas['Currency'] = currency
        df_currency_with_whole_datas['Interval'] = interval

        # Set columns `Close_time`
        df_currency_with_whole_datas = df_currency_with_whole_datas.merge(df_currency[['Close_time', 'Source']], on='Open_time', how='left')

        # Calculate the `Close_time` where a value is missing
        df_currency_with_whole_datas['Open_time'] = df_currency_with_whole_datas.index
        df_currency_with_whole_datas['Close_time'] = df_currency_with_whole_datas.apply(
            lambda row: self.calculate_close_time(row) if pd.isna(row['Close_time']) else row['Close_time'],
            axis=1
        )
        #df_currency_with_whole_datas.drop(columns=['Open_time'], inplace=True)

        df_currency_with_whole_datas['Number_of_trades'] = df_currency_with_whole_datas['Number_of_trades'].astype(int)

        # Calculate the `Close_time` where a value is missing
        df_currency_with_whole_datas['is_return'] = df_currency_with_whole_datas['Source'].fillna(1)
        df_currency_with_whole_datas['Source'] = df_currency_with_whole_datas['Source'].fillna('pandas_interpolate')

        return df_currency_with_whole_datas[df_currency_with_whole_datas['is_return'] == 1].drop(columns=['is_return'])


    def insert_into_db(self, df_currency, currency, interval):
        """
            Insert data into ch
        """
        data_tuples = [list(x) for x in df_currency[column_names].to_numpy()]
        insert_data(self.db_session, self.database, currency, interval, data_tuples)

        """
        # Add gotten data into CH
        self.db_session.insert(f"{self.database}{interval}.{currency.lower()}", 
                               data_tuples,
                               column_names=['Open', 'High', 'Low', 'Close', 'Volume',
                                             'Quote_asset_volume', 'Number_of_trades',
                                             'Taker_buy_base_asset_volume',
                                             'Taker_buy_quote_asset_volume', 'Currency', 'Interval', 'Close_time', 'Source', 'Open_time'])
        """