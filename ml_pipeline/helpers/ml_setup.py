from database.clickHouseConnection import ClickHouseConnection



class ConfigurationBuilder:
    """
        A builder class for setting various configuration parameters using a fluent interface pattern.
        Each setter method allows chaining by returning the instance itself.
    """

    def __init__(self):
        # ClickHouse connection object
        self.clickhouse_conn = ClickHouseConnection()

        self.db_session = None
        self.start = None
        self.end = None
        self.currency = None
        self.interval = None

    def set_db_session(self, db_session=None):
        self.db_session = db_session
        return self

    def set_database(self, database=None):
        self.database = database
        return self

    def set_modelname(self, modelname=None):
        self.modelname = modelname
        return self

    def set_model(self, model=None):
        self.model = model
        return self

    def set_start(self, start=None):
        self.start = start
        return self

    def set_end(self, end=None):
        self.end = end
        return self

    def set_currency(self, currency=None):
        self.currency = currency.lower()
        return self

    def set_interval(self, interval=None):
        self.interval = interval
        return self

