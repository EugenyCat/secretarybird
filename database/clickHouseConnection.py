import clickhouse_connect
import os
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
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
        self.session = None
        self.engine = None

        self.__host = host or os.getenv('CLICKHOUSE_HOST')
        self.__port = port or os.getenv('CLICKHOUSE_PORT')
        self.__user = user or os.getenv('CLICKHOUSE_USER')
        self.__password = password or os.getenv('CLICKHOUSE_PASSWORD')
        self.__timeout = 300


    def set_connection_timeout(self, timeout=1500):
        self.__timeout = timeout


    def get_client_session(self):
        """
            Get a ClickHouse client session, creating one if it does not already exist.
        """
        if self.client is None:
            self.client = clickhouse_connect.get_client(
                host=self.__host,
                port=self.__port,
                user=self.__user,
                password=self.__password,
                send_receive_timeout=self.__timeout
            )
        return self.client


    def get_sqlalchemy_session(self, database_name):
        """
            Get an SQLAlchemy session, creating one if it does not already exist.
        """
        if self.engine is None:
            # SQLAlchemy params
            database_url = f"clickhouse+http://{self.__user}:{self.__password}@{self.__host}:{self.__port}"
            self.engine = create_engine(database_url)
            self.session = sessionmaker(bind=self.engine)

        return self.session()


    def close_session(self):
        """
            Close the ClickHouse client session if it is open and clean up the client reference.
        """
        if hasattr(self, 'client') and self.client:
            self.client.close()
            del self.client
            self.client = None
        if self.engine:
            self.engine.dispose()
            self.engine = None
        self.session = None