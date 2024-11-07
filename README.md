# secretarybird

The project is designed to collect, store, and analyze time series data of securities and various currencies to forecast future prices.

## Table of Contents:

### ETL Process
The process extracts data from the Binance API (support for **yfinance**, **Tinkoff API**, **Alpha Vantage API**, and news sources is planned in the future) and stores it in the **ClickHouse** database. The data extractor runs hourly using **Airflow**. Additionally, a procedure for data quality assessment, missing data completion, backups, and Telegram notifications for unforeseen situations has been implemented. All code is written according to design patterns, enhancing its quality and maintainability.

### Machine Learning and Neural Networks
Development of machine learning and neural network models for generating forecasts is currently in progress.

### Installation
This section describes the process of installing and launching the project on a local computer.

---

### Cloning the Repository
To get started, you need to clone the **"secretarybird"** repository. Open a terminal and run the following command:

```bash
git clone https://github.com/EugenyCat/secretarybird.git
``` 

This will create a local copy of the repository on your computer.

### Creating a Virtual Environment
It is recommended to use a virtual environment to manage the project's dependencies. Navigate to the project folder and execute the following steps to create a virtual environment:

```bash
cd secretarybird
pip install virtualenv
python -m venv .venv
```
This will create a virtual environment folder named ".venv" in the project's root directory.

### Installing Dependencies
After creating the virtual environment, you need to activate it and install the dependencies. Execute the following commands:

### Activating the Virtual Environment

```bash
source .venv/bin/activate  # for Windows use ".venv\Scripts\activate"
```

### Installing Dependencies

After activating the virtual environment, install the project dependencies listed in `requirements.txt` by executing the following command:

```bash
pip install -r requirements.txt
```

This command will read the `requirements.txt` file and install all the necessary packages in the virtual environment.

### Running Docker Containers
While in the project's root folder, open a terminal and run the following command (make sure Docker is installed):

```bash
docker-compose --env-file .env.docker up
```

This will start and configure all necessary `Docker` containers: `Airflow`, `ClickHouse`, and `Grafana`.

### Configuring the .env File
Create or request a `.env` file with the required credentials:

```bash
# Airflow Credentials
AIRFLOW_UID=
_AIRFLOW_WWW_USER_USERNAME=
_AIRFLOW_WWW_USER_PASSWORD=
AIRFLOW__WEBSERVER__SECRET_KEY=

# PostgreSQL Credentials for Airflow
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=

# ClickHouse Connection
CLICKHOUSE_HOST=
CLICKHOUSE_PORT=
CLICKHOUSE_USER=
CLICKHOUSE_PASSWORD=

# Grafana Credentials
GRAFANA_URL=
GF_SECURITY_ADMIN_USER=
GF_SECURITY_ADMIN_PASSWORD=

# Telegram AirflowAlertBot Info and Tokens
NAME_TELEGRAM_ALERT_BOT=
USERNAME_TELEGRAM_ALERT_BOT=
TOKEN_TELEGRAM_ALERT_BOT=
CHAT_ID_TELEGRAM_ALERT_BOT=

# Binance API Info
BINANCE_API_LINK=
```

### Accessing Airflow
To access `Airflow` and run `ETL processes`, navigate to the following link (for local deployment):

```bash
http://localhost:8080
```
<b>Author</b>: `teapartygirl`