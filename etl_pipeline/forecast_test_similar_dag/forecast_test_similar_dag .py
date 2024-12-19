from etl_pipeline.ml_workflows.modelAutomationManager import ModelAutomationManager
from etl_pipeline.etl_manager.forecastManager import ForecastManager

mam = ModelAutomationManager()

ml_params = mam.ml_params
#print(ml_params)

for source_name in ml_params['ts_source_name']:
    #print(source_name)
    manager = ForecastManager(source_name)
    train_table_names = manager.get_ts_tables()
    #print(train_table_names)
    for table in train_table_names:
        for model in ml_params['model_names']:
            params = {
                'model_name': model,
                'ts_table_name': table,
                'database': manager.database
            }
            res, err = mam.run(params)
            print(res or err)