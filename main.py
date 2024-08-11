from pipeline.etl.ETLBinance import ETLBinance

#if __name__=='__main__':

input_params = {
        'start': '2018-04-01 00:00',
        'end': '2018-07-10 20:00',
        'currency': 'BTCUSDT',
        'interval': '1h'
    }

obj = ETLBinance()
print(obj.run_etl(input_params))