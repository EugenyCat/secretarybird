from pipeline.helpers.etl_setup import ConfigurationBuilder
from datetime import datetime
from dateutil.relativedelta import relativedelta


class ETLUtils(ConfigurationBuilder):
    """
        Utility class for various ETL helper functions.
    """

    @staticmethod
    def count_months(start_date, end_date):
        """
            Calculate the number of months between two dates (static method).
        """
        return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1


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
        return [tuple(int(x.timestamp() * 1000) for x in el_interval) for el_interval in intervals]


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
                (self.interval == '1h' and total_days < 19)
                or
                (self.interval == '12h' and total_days < 250)
                or
                (self.interval == '1d' and total_days < 500)
                or
                (self.interval == '3d' and total_days < 1500)
                or
                (self.interval == '1w' and total_days < 3500)
                or
                (self.interval == '1M' and total_years < 40)
        ):
            inters = [
                (self.start, self.end)
            ]
        else: # when date range is big and exceeds 500 data points
            if self.interval == '12h':
                rel_data = relativedelta(months=7) + relativedelta(day=31)
            elif self.interval == '1d':
                rel_data = relativedelta(years=1) + relativedelta(months=3) + relativedelta(day=31)
            elif self.interval == '3d':
                rel_data = relativedelta(years=3) + relativedelta(months=9) + relativedelta(day=31)
            elif self.interval == '1w':
                rel_data = relativedelta(years=9) + relativedelta(day=31)
            elif self.interval == '1M':
                rel_data = relativedelta(years=40) + relativedelta(day=31)
            else:   # elif interval == '1h'
                rel_data = relativedelta(days=19)

            inters = []

            # Calculate tuples represent an interval (interval_start, interval_end)
            current_date = self.start
            while current_date < self.end - rel_data:
                inters.append(
                    (
                        current_date,
                        (current_date + rel_data).replace(hour=23, minute=59, second=59)
                    )
                )
                current_date = (current_date + rel_data + relativedelta(days=1)).replace(hour=0, minute=0, second=0)

            # Add the last interval
            inters.append(
                (
                    current_date,
                    self.end
                )
            )

        return inters



"""
start = datetime.strptime('2024-08-01 14:00', '%Y-%m-%d %H:%M')
end = datetime.strptime('2024-09-30 23:00', '%Y-%m-%d %H:%M')
currency = 'BTCUSDT'
interval= '1h'

a = (ETLUtils()
                .set_currency(currency)
                .set_interval(interval)
                  .set_start(start)
                .set_end(end)
            )
inters = a.get_list_of_intervals()
print(inters)
"""