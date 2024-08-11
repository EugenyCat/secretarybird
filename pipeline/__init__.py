from database.ClickHouseConnection import ClickHouseConnection
from database.ClickHouseFuncs import get_extreme_date
from dateutil.relativedelta import relativedelta
import datetime
import os


class ExtractBase:

    def __init__(self):

        self.ch_connection_obj = ClickHouseConnection()
        self.null_datetime = datetime.datetime(1970, 1, 1, 0, 0)
        self.null_datetime_ms = int(datetime.datetime(1970, 1, 1, 0, 0).timestamp() * 1000)


    def count_months(self, start_date, end_date):
        return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1


    def get_list_of_intervals(self, interval, start, end):
        """

        :param start: first interval data of extracted data
        :param end: last interval data of extracted data
        :return: list of tuples , where each tuple is month's interval from [start, end]
        """

        #total_months = self.count_months(start, end)
        total_days = (end - start).days
        total_year = end.year - start.year

        if (
                (interval in ['1h'] and total_days < 21)
                or
                (interval in ['12h'] and total_days < 250)
                or
                (interval in ['1d'] and total_days < 500)
                or
                (interval in ['3d'] and total_days < 1500)
                or
                (interval in ['1w'] and total_days < 3500)
                or
                (interval in ['1M'] and total_year < 40)
        ):
            inters = [
                (start, self.null_datetime, self.null_datetime, end)
            ]
        else:
            if interval == '12h':
                rel_data = relativedelta(months=14) + relativedelta(day=31)
            elif interval == '1d':
                rel_data = relativedelta(years=2) + relativedelta(day=31)
            elif interval == '3d':
                rel_data = relativedelta(years=8) + relativedelta(day=31)
            elif interval == '1w':
                rel_data = relativedelta(years=19) + relativedelta(day=31)
            elif interval == '1w':
                rel_data = relativedelta(years=80) + relativedelta(day=31)
            else:   # elif interval == '1h'
                rel_data = relativedelta(days=31)

            inters = []

            current_date = start
            while current_date < end.replace(day=1, hour=0, minute=0, second=0) - rel_data:
                inters.append(
                    self.split_into_intervals(
                        current_date,
                        (current_date + rel_data).replace(hour=23, minute=59, second=59)
                    )
                )
                current_date = (current_date + rel_data + relativedelta(days=1)).replace(hour=0, minute=0, second=0)

            inters.append(
                self.split_into_intervals(
                    current_date,
                    end
                )
            )

        """
        for dat in inters:
            print(
                f"{dat[0].strftime('%Y-%m-%d %H:%M:%S')} - {dat[1].strftime('%Y-%m-%d %H:%M:%S')} - {dat[2].strftime('%Y-%m-%d %H:%M:%S')} - {dat[3].strftime('%Y-%m-%d %H:%M:%S')}")
        """

        return inters


    def split_into_intervals(self, start, end):
        return (
            start,
            (start + relativedelta(days=(end - start).days // 2)).replace(hour=23, minute=59, second=59),
            (start + relativedelta(days=(end - start).days // 2 + 1)).replace(hour=0, minute=0, second=0),
            end
        )


    def define_start_time(self, database, currency, interval):
        try:
            min_data = get_extreme_date(self.db_session, database, currency, interval)
            return min_data.strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            print(f"Error of getting the max data from {database}{interval}.{currency.lower()}: {e}")

        return os.getenv('DEFAULT_DATA_START_LOAD')


    def convert_intervals_into_ms(self, intervals):
        return [tuple(int(x.timestamp() * 1000) for x in el_interval) for el_interval in intervals]