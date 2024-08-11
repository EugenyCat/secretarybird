import clickhouse_connect
import os
from dotenv import load_dotenv
load_dotenv()

CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST')
CLICKHOUSE_PORT = os.getenv('CLICKHOUSE_PORT')
CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER')
CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD')

class ClickHouseConnection:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, user=CLICKHOUSE_USER, password=CLICKHOUSE_PASSWORD):
        self.client = clickhouse_connect.get_client(host=host, port=port, user=user, password=password)

    def get_session(self):
        return self.client

    def close_session(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()
            del self.client
            self.client = None