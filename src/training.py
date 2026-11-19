import pandas as pd
import numpy as np
import pickle
from config import exogenous_variables, target, target_diff, test_split, DATA_DIR, MODEL_DIR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error , mean_absolute_error , mean_absolute_percentage_error          
pd.options.display.float_format = '{:,.2f}'.format
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings



def data_preparation(df, selected_cols):
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    X_columns = df.drop([target, target_diff], axis = 1)

    exog_train = X_columns.iloc[5:test_split][selected_cols]
    exog_test = X_columns.iloc[test_split:][selected_cols]

    y_train = df.iloc[5:test_split][target_diff]
    y_test = df.iloc[test_split:][target_diff]

    return exog_train, exog_test, y_train, y_test


def model(df,selected_cols ):
    exog_train, exog_test, y_train, y_test = data_preparation(df, selected_cols)

    best_param = (0, 0, 0)
    best_param_seasonal = (1, 0, 0, 12)

    model = SARIMAX(y_train,
                    exog=exog_train, 
                    order=best_param,
                    seasonal_order=best_param_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    results = model.fit()

    y_test_pred = results.predict(
                            start=exog_test.index[0],
                            end=exog_test.index[-1],
                            exog=exog_test
                                )
    test_rmse = round(root_mean_squared_error(y_test, y_test_pred), 5 )
    print(f"RMSE: {test_rmse}")

    anchor =  df[target].iloc[-13] 

    df_prection = pd.DataFrame({
        "date":y_test_pred.index,
        "unemployment_rate_fcst":anchor + y_test_pred.cumsum().values
        })

    df_final = df.merge(df_prection, on= 'date', how = 'left')

    return  df_final , results   

if __name__ == "__main__":
    # path = '../datasets/'
    df = pd.read_csv(DATA_DIR / "dataset_training.csv")
    print("funziona 1")
    # df = pd.read_csv(f"{path}/dataset_training.csv")
    df_final, results = model(df, exogenous_variables)
    print("funziona 2")
    # selected_cols = exogenous_variables

    with open(MODEL_DIR /  'sarimax_unemployment_12m.pkl', 'wb') as f:
        pickle.dump({'model': results, 'selected_cols': exogenous_variables}, f)

    print("model DONE")
    df_final.to_csv(DATA_DIR /  "dataset_predicion.csv")
    print("prediction DONE")

