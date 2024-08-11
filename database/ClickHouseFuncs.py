import logging
from functools import wraps
import time
import pandas as pd


valid_amount_of_rows = {
            "1h": 24,
            "12h": 2,
            "1d": 1,
            "3d": [9, 10],
            "1w": 4,
            "1M": 1
}

column_names=['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time',
                      'Quote_asset_volume', 'Number_of_trades', 'Taker_buy_base_asset_volume',
                      'Taker_buy_quote_asset_volume', 'Currency', 'Interval', 'Source']

def retry(retries=5, backoff_factor=0.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        time.sleep(backoff_factor + (0.5 * attempt))
                    else:
                        raise e

        return wrapper

    return decorator


@retry()
def insert_data(db_session, database, currency, interval, data_tuples): # TODO : add a good name
    """
        Insert data into CH ....
    """
    db_session.insert(
        f'{database}{interval}.{currency.lower()}',
        data_tuples,
        column_names=column_names
    )


@retry()
def remove_data(db_session, database, currency, interval, start, end):
    """
        Remove data for the period
    """
    remove_request = f"ALTER TABLE {database}{interval}.{currency.lower()} DELETE WHERE Open_time >= '{start.strftime('%Y-%m-%d %H:%M:%S')}' AND Open_time <= '{end.strftime('%Y-%m-%d %H:%M:%S')}'"
    db_session.query(remove_request)


@retry()
def create_table(db_session, database, currency, interval):
    """
        Create table if not exists
    """
    # Create table if not exists
    create_tab_request = f"CREATE TABLE IF NOT EXISTS {database}{interval}.{currency.lower()}  ( \
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

    db_session.query(create_tab_request)


@retry()
def create_db(db_session, database, interval):
    """
        Create db
    """
    create_db = f"CREATE DATABASE IF NOT EXISTS {database}{interval}"
    db_session.query(create_db)


@retry()
def get_extreme_date(db_session, database, currency, interval, method='max'):
    """
        Define last data from db
    """
    end_time = db_session.query(f"""
                SELECT {method}(Open_time) 
                FROM {database}{interval}.{currency.lower()}
            """).result_rows[0][0]

    return end_time


def get_where_condition(interval):
    """
        Generate WHERE condition depends on `interval`
    """
    row_num_coef = valid_amount_of_rows[f'{interval}']

    if interval in ['1h', '12h', '1d']:
        where_condition = f"""
                        WHERE   (rows_amount < {row_num_coef * 31} AND month_name in [1, 3, 5, 7, 8, 10, 12]) 
                                OR 
                                (rows_amount < {row_num_coef * 30} AND month_name in [4, 6, 9, 11])
                                OR 
                                (rows_amount < {row_num_coef * 29} AND month_name = 2 AND is_leap = 1)
                                OR 
                                (rows_amount < {row_num_coef * 28} AND month_name = 2 AND is_leap = 0)
                """
    elif interval == '3d':
        where_condition = f"""
                        WHERE   (rows_amount < {row_num_coef[0]} AND month_name = 2) 
                                OR 
                                (rows_amount < {row_num_coef[1]} AND month_name != 2)
                """
    elif interval in ['1w', '1M']:
        where_condition = f"""
                        WHERE   rows_amount < {row_num_coef}
                """
    else:
        raise Exception("SOMETHING WRONG! BROKE THE REQUEST!")

    return where_condition


@retry()
def get_lacks(db_session, database, currency, interval):
    """
        Define the current lacks in CH db
    """
    where_condition = get_where_condition(interval)

    # TODO: hide this request
    get_records_count_request = f"""
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
                        FROM {database}{interval}.{currency.lower()}
                        WHERE Open_time < toStartOfMonth(today())
                                and Open_time >= (
                                    SELECT toStartOfMonth(addMonths(MIN(Open_time), 1))
                                    FROM {database}{interval}.{currency.lower()}
                                )
                        GROUP BY month_where_lacks, month_name, is_leap
                        ORDER BY rows_amount
                    )
                    {where_condition}
            """

    ch_response_df = pd.DataFrame(
        db_session.query(get_records_count_request).result_rows,
        columns=['month_where_lacks', 'interval', 'currency', 'rows_amount']
    )

    return ch_response_df


@retry()
def get_amount_of_rows_for_currency_and_interval(db_session, database, currency, interval):
    # Calculate the amount of rows for the currency and interval
    query_count_open_time = f"""
                        SELECT count(Open_time) 
                        FROM {database}{interval}.{currency.lower()}
                    """
    return db_session.query(query_count_open_time).result_rows[0][0]


@retry()
def define_doubles_in_tables(db_session, database, currency, interval):
    """
        Send a query to CH and get all doubles data over currency, interval
    """
    # Get data from db
    ch_response = db_session.query(f"""
                                SELECT Open_time, COUNT(*) AS count
                                FROM {database}{interval}.{currency.lower()}
                                GROUP BY Open_time
                                HAVING count > 1
                            """)

    # Create a df
    df_currency = pd.DataFrame(ch_response.result_rows, columns=ch_response.column_names)
    logging.info(f'==============={df_currency}')
    return df_currency


@retry()
def remove_doubles(db_session, database, currency, interval):
    """
        Rewrite the table with uniq values
    """
    # Drop table
    drop_request = f"""
                    DROP TABLE IF EXISTS {database}{interval}.{currency.lower()}_no_duplicates;
                """
    db_session.query(drop_request)

    # Create table
    create_tab_request = f"CREATE TABLE {database}{interval}.{currency.lower()}_no_duplicates  ( \
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

    db_session.query(create_tab_request)

    # Insert table
    insert_tab_request = f"""
        INSERT INTO {database}{interval}.{currency.lower()}_no_duplicates
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
        FROM {database}{interval}.{currency.lower()}
        GROUP BY 
        Open_time
        """

    db_session.query(insert_tab_request)

    # Rename table
    rename_request = f"""
        RENAME TABLE {database}{interval}.{currency.lower()} TO {database}{interval}.{currency.lower()}_old, {database}{interval}.{currency.lower()}_no_duplicates TO {database}{interval}.{currency.lower()};
    """
    db_session.query(rename_request)

    # Drop table
    drop_request = f"""
                DROP TABLE {database}{interval}.{currency.lower()}_old;
            """
    db_session.query(drop_request)


@retry()
def get_data(db_session, database, currency, interval):
    """
        Get data from db
    """
    ch_response = db_session.query(f"""
                                SELECT *
                                FROM {database}{interval}.{currency.lower()}
                            """)

    return pd.DataFrame(ch_response.result_rows, columns=ch_response.column_names)