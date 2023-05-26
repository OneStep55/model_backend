import json
import pickle
import pandas as pd
__periods = None
__data_columns = None
__model = None
__df = None



def read_data():
    global __df
    __df = pd.read_csv("./artifacts/new_data.csv")
    __df['Datetime'] = pd.to_datetime(__df['Datetime']) 
    __df = __df.set_index('Datetime')
    
   

def load_saved_artifacts():
    print("loading artifacts starting")
    global __data_columns
    global __periods
    global __model
    global __res

    with open("./artifacts/columns.json", "r") as f:
         __data_columns = json.load(f)['data_columns']
         __periods = __data_columns[2:]
    
    with open("./artifacts/electrcity_pred_model.pickle", "rb") as f:
        __model = pickle.load(f)
    print("loading artifacts done")

def create_featureAkm(df):
    """
    Create time series featuAkm based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    # df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    # df['dayofyear'] = df.index.dayofyear
    # df['dayofmonth'] = df.index.day
    # df['weekofyear'] = df.index.isocalendar().week
    return df

def create_future_df(startDate, EndDate):
    load_saved_artifacts()
    FEATURES = ['hour', 'month', 'dayofweek', 'year',
            'lag1','lag2','lag3']
    TARGET = 'Consumption'
    global __df
    __df = pd.read_csv("./artifacts/new_data.csv")
    __df['Datetime'] = pd.to_datetime(__df['Datetime']) 
    __df = __df.set_index('Datetime')
    future = pd.date_range(startDate, EndDate, freq='1h')
    future_df = pd.DataFrame(index=future)
    future_df['isFuture'] = True
    __df['isFuture'] = False
    df_and_future = pd.concat([__df, future_df])
    df_and_future = create_featureAkm(df_and_future)
    future_w_featuAkm = df_and_future.query('isFuture').copy()
    future_w_featuAkm['pred'] = __model.predict(future_w_featuAkm[FEATURES])
    new_df = future_w_featuAkm.reset_index()[['index', 'pred']]
    new_df = new_df.rename(columns={'index': 'datetime'})
    __res = new_df['pred'].sum()
    return __res




if __name__ == "__main__":
    print("Main")
