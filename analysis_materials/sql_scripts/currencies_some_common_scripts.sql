USE default;

CREATE DATABASE IF NOT EXISTS CRYPTO_1h;

CREATE DATABASE IF NOT EXISTS MODEL_REGISTRY;

DROP DATABASE CRYPTO;


--SHOW DATABASES;


-- creation of a table for storing time series on bitcoin
CREATE TABLE IF NOT EXISTS CRYPTO_1h.bitcoin (
	Currency String,
	Interval String,
    Open_time DateTime,
    Open Float64,
    High Float64,
    Low Float64,
    Close Float64,
    Volume Float64,
    Close_time DateTime,
    Quote_asset_volume Float64,
    Number_of_trades UInt32,
    Taker_buy_base_asset_volume Float64,
    Taker_buy_quote_asset_volume Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(Open_time)
ORDER BY (Open_time);


--DROP TABLE IF EXISTS my_database.my_table;
ALTER TABLE CRYPTO_1h.bitcoin ADD COLUMN Source String DEFAULT 'binance_api';


SELECT * from CRYPTO_1h.btcusdt;


backup database CRYPTO_1h TO Disk('backups', 'CRYPTO_1h-backup.zip')

backup database CRYPTO_1h TO Disk('backups', 'CRYPTO_1h_backup_2024-07-15__15-34.zip')




BACKUP DATABASES TO Disk('backups', 'all_databases_backup.zip');

SHOW DATABASES



RESTORE DATABASES CRYPTO FROM Disk('backups', 'CRYPTO_1h-backup.zip');

RESTORE DATABASE CRYPTO_1h FROM Disk('backups', 'CRYPTO_1h-backup.zip')


RESTORE DATABASE CRYPTO_1h FROM Disk('backups', 'CRYPTO_1h-backup.zip')



SHOW DATABASES LIKE 'CRYPTO_1h'

SHOW DATABASES





SELECT count(*) from CRYPTO_1w.ethusdt  ;

SELECT * from CRYPTO_1w.ethusdt

WHERE Open_time >= '2021-02-01 00:00:00'
AND Open_time <= '2021-03-01 00:00:00'


WITH min_data AS (
	SELECT toStartOfMonth(min_data) AS min_month_data, day(toLastDayOfMonth(min_data)) - day(min_data) + 1 AS count_for_min_data
	FROM (
		SELECT min(Open_time) AS min_data
		FROM CRYPTO_.btcusdt_1h
	)
)
SELECT *
FROM (
	SELECT 
	    toStartOfMonth(Open_time) AS first_month_day,
	    MONTH(Open_time) AS month_name,
	    IF (MONTH(Open_time) = 2 AND DAY(toLastDayOfMonth(Open_time)) = 28, 0,
	    	IF (MONTH(Open_time) = 2 AND DAY(toLastDayOfMonth(Open_time)) = 29, 1,
				NULL 
	    )) AS is_leap,
	    count(*) AS record_count
	FROM CRYPTO_.btcusdt_1h
	WHERE Open_time < toStartOfMonth(today())
	and Open_time >= (
			SELECT toStartOfMonth(addMonths(MIN(Open_time), 1))
            FROM CRYPTO_.btcusdt_1h
    )
	GROUP BY first_month_day, month_name, is_leap
	ORDER BY record_count
) tab
WHERE 
	(record_count < 744 AND month_name in [1, 3, 5, 7, 8, 10, 12]) 
	OR 
	(record_count < 720 AND month_name in [4, 6, 9, 11])
	OR 
	(record_count < 696 AND month_name = 2 AND is_leap = 1)
	OR 
	(record_count < 672 AND month_name = 2 AND is_leap = 0)
	
	

Январь (1)
Март (3)
Май (5)
Июль (7)
Август (8)
Октябрь (10)
Декабрь (12)

SELECT 
    *
FROM CRYPTO_3d.xrpusdt b
WHERE Open_time >= '2018-06-28 00:00:00' 
  AND Open_time < '2018-07-01 00:00:00'
  
  
  
  
SELECT *  from CRYPTO_1h.btcusdt s 

WHERE Open_time >= '2023-03-01 00:00:00'
AND Open_time < '2023-04-01 00:00:00'

ALTER TABLE CRYPTO_.btcusdt_1h DELETE WHERE Open_time >= '2021-03-01 00:00:00'
AND Open_time < '2022-04-01 00:00:00';

ALTER TABLE CRYPTO.btcusdt_1h DELETE WHERE Open_time >= '2023-10-05 00:00:00';

ALTER TABLE CRYPTO.btcusdt_1h DELETE WHERE Open_time = '2024-09-05 14:00:00';


--INSERT INTO CRYPTO_1h.xrpusdt (Currency, Interval, Open_time, Open, High, Low, Close, Volume, Close_time, Quote_asset_volume, Number_of_trades, Taker_buy_base_asset_volume, Taker_buy_quote_asset_volume, Source)
--VALUES ('XRPUSDT', '1h', '2018-06-01 00:00:00', 0.6114, 0.61549, 0.60901, 0.60901, 1111446.01, '2018-06-01 00:59:59', 679763.6995305, 919, 710866.17, 434706.7065649, 'binance_api')



SELECT Open_time, COUNT(*) AS count  from CRYPTO_12h.btcusdt
WHERE Open_time >= '2018-02-01 00:00:00'
AND Open_time < '2018-03-01 00:00:00'
GROUP BY Open_time
HAVING count > 1


SELECT max(Open_time), MIN(Open_time), count(*)  from CRYPTO.btcusdt_1h_raw bh  ;

SELECT max(Open_time), MIN(Open_time), count(*)  from CRYPTO.xrpusdt_1h_raw xhr   ;

SELECT max(Open_time), MIN(Open_time), count(*)  from CRYPTO.btcusdt_1h bh  ;

DROP TABLE CRYPTO.btcusdt_1h;


SELECT * from
CRYPTO.btcusdt_1h WHERE Open_time = '2024-09-05 14:00:00';



SELECT DISTINCT 
    splitByString('_', name)[1] AS v_currency
FROM system.tables
WHERE database = 'CRYPTO'
AND name LIKE '%_%';


SELECT max(Open_time), MIN(Open_time) from
CRYPTO.xrpusdt_1h





select tpp.ts_id, tpp.main_period , tpp.period_candidates  from ConfigKernel.ts_preprocessing_properties tpp 



