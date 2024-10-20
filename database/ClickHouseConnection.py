import clickhouse_connect
import os
from dotenv import load_dotenv
load_dotenv()


class ClickHouseConnection:
    _instance = None

    def __new__(cls):
        """
            Ensure only one instance of ClickHouseConnection is created using the Singleton pattern.
        """
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance


    def __init__(self, host=None, port=None, user=None, password=None):
        """
            Initialize the session manager with connection details, using environment variables if not provided.
        """
        self.client = None
        self.__host = host or os.getenv('CLICKHOUSE_HOST')
        self.__port = port or os.getenv('CLICKHOUSE_PORT')
        self.__user = user or os.getenv('CLICKHOUSE_USER')
        self.__password = password or os.getenv('CLICKHOUSE_PASSWORD')


    def get_session(self):
        """
            Get a ClickHouse client session, creating one if it does not already exist.
        """
        if self.client is None:
            self.client = clickhouse_connect.get_client(
                host=self.__host,
                port=self.__port,
                user=self.__user,
                password=self.__password
            )
        return self.client


    def close_session(self):
        """
            Close the ClickHouse client session if it is open and clean up the client reference.
        """
        if hasattr(self, 'client') and self.client:
            self.client.close()
            del self.client
            self.client = None