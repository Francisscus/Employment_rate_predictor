
import pandas as pd
import requests
from io import StringIO
from functools import reduce
import time 
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from config import exogenous_variables, DATA_DIR

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')



# start_year_month = '2013-01'
# end_year_month = '2026-07'


# def API_calling(dataflow, start_date = start_year_month, end_date = end_year_month):
def API_calling(dataflow, start_date, end_date):
    # API endpoint and dataset
    base = "https://esploradati.istat.it/SDMXWS/rest/data"
    # dataflow = "IT1,150_875"
    dataflow = dataflow

    # Query parameters
    params = {
        "startPeriod": f"{start_date}",
        "endPeriod": f"{end_date}",
        #"POSIZ_PROF": "1", #Dipedenti
        #"PERM_TEMP_EMPLOYEES": ""
        "format": "CSV"  # request CSV
    }

    # Headers (fixed syntax)
    headers = {"Accept": "text/csv"}

    # Build URL
    url = f"{base}/{dataflow}"

    # Send GET request
    response = requests.get(url, params=params, headers=headers, verify=False)

    # Check for errors
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    # Read CSV into pandas directly
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)

    return df



def extract_emplyoment_data(start_year_month, end_year_month):
      df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,150_875_DF_DCCV_OCCUPATIMENS1_1")

      #"POSIZ_PROF"--> #'posizione professionale': # 1: Dipedenti, # 2: Indipendenti,# 9: Totale
      #"PERM_TEMP_EMPLOYEES" --> 'Categoria occupazione dipendente' #1: Tempo determinato, 2: Tempo indeterminato,9: Totale

      last_edition = df['EDITION'].unique().max()
      df = df[df['EDITION'] == last_edition]

      df = df[(df['ADJUSTMENT'] == 'N') #prendo  solo le righe che NON hanno adjustments
            & (df['POSIZ_PROF']==1)  # prendo solo i lavoratori Dipendenti
            & (df['PERM_TEMP_EMPLOYEES']!=9)] # prendo solo il valore totale

      # df['PERM_TEMP_EMPLOYEES'].replace({1:'fixed_term', 2 : 'permanent'}, inplace = True)
      df['PERM_TEMP_EMPLOYEES'] = df['PERM_TEMP_EMPLOYEES'].replace({1:'fixed_term', 2 : 'permanent'})
      df_employment_pivoted = df.pivot(index='TIME_PERIOD', columns='PERM_TEMP_EMPLOYEES', values=['OBS_VALUE'])

      df_employment_pivoted = df_employment_pivoted.rename_axis(None)

      df_employment_pivoted.columns = df_employment_pivoted.columns.droplevel(0)

      df_employment_pivoted['date'] =  pd.to_datetime(df_employment_pivoted.index, format='%Y-%m')

      df_employment_pivoted = df_employment_pivoted.rename_axis(None, axis=1)

      df_employment_pivoted = df_employment_pivoted.reset_index()#
      df_employment_pivoted.drop('index', axis = 1, inplace = True)#

      return df_employment_pivoted



def extract_employment_rate(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,150_872_DF_DCCV_TAXOCCUMENS1_1")

    last_edition = df['EDITION'].unique().max() # this is necessary to take the most updated values
    df = df[df['EDITION'] == last_edition]

    df_employed_rate = df[(df.SEX==9) & (df.AGE.isin(['Y15-64','Y15-34'])) & (df.ADJUSTMENT == 'N')] # I select youth employment and full age range employment

    df_employed_rate_pivoted = df_employed_rate.pivot(index='TIME_PERIOD', columns='AGE', values=['OBS_VALUE'])

    df_employed_rate_pivoted = df_employed_rate_pivoted.rename_axis(None)

    df_employed_rate_pivoted.columns = df_employed_rate_pivoted.columns.droplevel(0)

    df_employed_rate_pivoted['date'] =  pd.to_datetime(df_employed_rate_pivoted.index, format='%Y-%m')

    df_employed_rate_pivoted = df_employed_rate_pivoted.rename_axis(None, axis=1)

    df_employed_rate_pivoted.rename(columns={'Y15-64': 'employment rate (Y15-64)', 'Y15-34':  'youth employment rate (Y15-34)'}, inplace = True)

    df_employed_rate_pivoted = df_employed_rate_pivoted.reset_index()#
    df_employed_rate_pivoted.drop('index', axis = 1, inplace = True)#

    return df_employed_rate_pivoted



def extract_workforce_data(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,150_873_DF_DCCV_FORZLVMENS1_1")

    last_edition = df['EDITION'].unique().max() # this is necessary to take the most updated values
    df = df[df['EDITION'] == last_edition]

    df_workforce = df[(df.SEX==9) & (df.AGE=='Y15-64') & (df.ADJUSTMENT == 'N')]

    df_workforce_pivoted = df_workforce.pivot(index='TIME_PERIOD', columns='AGE', values=['OBS_VALUE'])

    df_workforce_pivoted = df_workforce_pivoted.rename_axis(None)

    df_workforce_pivoted.columns = df_workforce_pivoted.columns.droplevel(0)

    df_workforce_pivoted['date'] =  pd.to_datetime(df_workforce_pivoted.index, format='%Y-%m')

    df_workforce_pivoted = df_workforce_pivoted.rename_axis(None, axis=1)

    df_workforce_pivoted.rename(columns={'Y15-64': 'workforce (Y15-64)'}, inplace = True)

    df_workforce_pivoted = df_workforce_pivoted.reset_index()#
    df_workforce_pivoted.drop('index', axis = 1, inplace = True)#

    return df_workforce_pivoted



def activity_rate(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,150_876_DF_DCCV_TAXATVTMENS1_1")

    last_edition = df['EDITION'].unique().max() # this is necessary to take the most updated values
    df = df[df['EDITION'] == last_edition]

    df_activity_rate = df[(df.SEX==9) & (df.AGE=='Y15-64') & (df.ADJUSTMENT == 'N')]

    df_activity_rate_pivoted = df_activity_rate.pivot(index='TIME_PERIOD', columns='AGE', values=['OBS_VALUE'])

    df_activity_rate_pivoted = df_activity_rate_pivoted.rename_axis(None)

    df_activity_rate_pivoted.columns = df_activity_rate_pivoted.columns.droplevel(0)

    df_activity_rate_pivoted['date'] =  pd.to_datetime(df_activity_rate_pivoted.index, format='%Y-%m')

    df_activity_rate_pivoted = df_activity_rate_pivoted.rename_axis(None, axis=1)

    df_activity_rate_pivoted.rename(columns={'Y15-64': 'activity rate (Y15-64)'}, inplace =True)

    df_activity_rate_pivoted = df_activity_rate_pivoted.reset_index()#
    df_activity_rate_pivoted.drop('index', axis = 1, inplace = True)#

    return df_activity_rate_pivoted




def unemployment_rate(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,151_874_DF_DCCV_TAXDISOCCUMENS1_1")

    last_edition = df['EDITION'].unique().max() # this is necessary to take the most updated values
    df = df[df['EDITION'] == last_edition]

    df_unemployment_rate = df[(df.SEX==9) & (df.AGE=='Y15-64') & (df.ADJUSTMENT == 'N')]

    df_unemployment_rate_pivoted = df_unemployment_rate.pivot(index='TIME_PERIOD', columns='AGE', values=['OBS_VALUE'])

    df_unemployment_rate_pivoted = df_unemployment_rate_pivoted.rename_axis(None)

    df_unemployment_rate_pivoted.columns = df_unemployment_rate_pivoted.columns.droplevel(0)

    df_unemployment_rate_pivoted['date'] =  pd.to_datetime(df_unemployment_rate_pivoted.index, format='%Y-%m')#

    df_unemployment_rate_pivoted = df_unemployment_rate_pivoted.rename_axis(None, axis=1)

    df_unemployment_rate_pivoted.rename(columns={'Y15-64': 'unemployment rate (Y15-64)'}, inplace = True)

    df_unemployment_rate_pivoted = df_unemployment_rate_pivoted.reset_index()#

    df_unemployment_rate_pivoted.drop('index', axis = 1, inplace = True)#

    return df_unemployment_rate_pivoted



def extract_inactives_data(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,152_879_DF_DCCV_INATTIVMENS1_1")

    last_edition = df['EDITION'].unique().max() # this is necessary to take the most updated values
    df = df[df['EDITION'] == last_edition]

    df_inactives = df[(df.SEX==9) & (df.AGE=='Y15-64') & (df.ADJUSTMENT == 'N')]

    df_inactives_pivoted = df_inactives.pivot(index='TIME_PERIOD', columns='AGE', values=['OBS_VALUE'])

    df_inactives_pivoted = df_inactives_pivoted.rename_axis(None)

    df_inactives_pivoted.columns = df_inactives_pivoted.columns.droplevel(0)

    df_inactives_pivoted['date'] =  pd.to_datetime(df_inactives_pivoted.index, format='%Y-%m')#

    df_inactives_pivoted = df_inactives_pivoted.rename_axis(None, axis=1)

    df_inactives_pivoted.rename(columns={'Y15-64': 'inactives (Y15-64)'}, inplace = True)

    df_inactives_pivoted = df_inactives_pivoted.reset_index()#

    df_inactives_pivoted.drop('index', axis = 1, inplace = True)#

    return df_inactives_pivoted



def industry_production_prices(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,145_360_DF_DCSC_PREZZPIND_1_4")

    df_industry_production_prices = df[(df['MARKET']=='T')  & (df['ECON_ACTIVITY_NACE_2007']=='0020')] [['TIME_PERIOD', 'OBS_VALUE']].reset_index().rename(columns={'TIME_PERIOD':'date', 'OBS_VALUE':'industry_production_prices'})
    df_industry_production_prices['date'] =  pd.to_datetime(df_industry_production_prices['date'], format='%Y-%m')#
    df_industry_production_prices.drop('index', axis=1, inplace= True)

    return df_industry_production_prices


def extract_services_production_prices(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,145_376_DF_DCSC_PREZPRODSERV_1_7")

    df_services_production_prices = df[(df['ECON_ACTIVITY_NACE_2007']=='H_N_XK')][['TIME_PERIOD','OBS_VALUE']].reset_index().rename(columns={'TIME_PERIOD':'date', 'OBS_VALUE':'services_production_prices'})
    df_services_production_prices['date'] = df_services_production_prices['date'].apply(lambda x: pd.Period(x, freq="Q").to_timestamp(how="start")).dt.strftime("%Y-%m-%d")
    df_services_production_prices['date'] =  pd.to_datetime(df_services_production_prices['date'], format='%Y-%m-%d')
    df_services_production_prices.drop('index', axis=1, inplace= True)

    return df_services_production_prices


def extract_companies_trust(start_year_month, end_year_month):
    df = API_calling(start_date = start_year_month, end_date = end_year_month, dataflow = "IT1,6_64_DF_DCSC_IESI_2")

    df_companies_trust = df[['TIME_PERIOD','OBS_VALUE']].reset_index().rename(columns={'TIME_PERIOD':'date', 'OBS_VALUE':'companies_trust'})
    df_companies_trust['date'] =  pd.to_datetime(df_companies_trust['date'], format='%Y-%m')
    df_companies_trust.drop('index', axis=1, inplace= True)

    return df_companies_trust



def get_fuel_prices(fuel_type: str, start_year: int):

    fuel_type_dict = {'gasoline': 1,
                        'diesel':   2,
                        'GPL':   5,
                        'diesel_heatining':  3}

    fuel_code = fuel_type_dict.get(fuel_type)    

    df = pd.read_csv(f'https://sisen.mase.gov.it/dgsaie/api/v1/monthly-prices/{fuel_code}/export?format=CSV&lang=it' )
    df["date"] = pd.to_datetime(df["ANNO"].astype(str) + "-" + df["CODICE_MESE"].astype(str) + "-01",  format="%Y-%m-%d")
    df[f"{fuel_type}_price"] = df["PREZZO"].str.replace(",", "", regex=False)
    df[f"{fuel_type}_price"] = df[f"{fuel_type}_price"].astype('float')
    df = df[df['ANNO']>=start_year][['date', f'{fuel_type}_price']].reset_index()
    df.drop('index', axis=1, inplace=True)

    return df



def retrieving_data( start_year_month , end_year_month ):

        df_employment_pivoted  = extract_emplyoment_data(start_year_month, end_year_month)
        print("DONE 1 df_employment_pivoted")

        df_employed_rate_pivoted  = extract_employment_rate(start_year_month, end_year_month)
        print("DONE 2 df_employed_rate_pivoted")  

        df_workforce_pivoted = extract_workforce_data(start_year_month, end_year_month)
        print("DONE 3 df_workforce_pivoted")  

        df_activity_rate_pivoted  = activity_rate(start_year_month, end_year_month)
        print("DONE 4 df_activity_rate_pivoted")          

        df_unemployment_rate_pivoted = unemployment_rate(start_year_month, end_year_month)
        print("DONE 5 df_unemployment_rate_pivoted")               

        df_inactives_pivoted = extract_inactives_data(start_year_month, end_year_month)
        print("DONE 6 df_inactives_pivoted")                       

        df_industry_production_prices = industry_production_prices(start_year_month, end_year_month)
        print("DONE 7 df_industry_production_prices")                       

        df_services_production_prices = extract_services_production_prices(start_year_month, end_year_month)
        print("DONE 8 df_services_production_prices")    

        df_companies_trust = extract_companies_trust(start_year_month, end_year_month)
        print("DONE 9 df_companies_trust")    


        year = datetime.strptime(start_year_month, '%Y-%m').year 

        df_gasoline_car_price = get_fuel_prices(fuel_type = "gasoline", start_year = year)
        df_diesel_price = get_fuel_prices(fuel_type = "diesel", start_year = year)
        df_GPL_price  = get_fuel_prices(fuel_type = "GPL", start_year = year)
        df_diesel_heatining_price = get_fuel_prices(fuel_type = "diesel_heatining", start_year = year)


        df1 = pd.merge(df_employment_pivoted, df_employed_rate_pivoted, on='date', how='outer')

        df2 = pd.merge(df1, df_workforce_pivoted, on='date', how='outer')

        df3 = pd.merge(df2, df_activity_rate_pivoted, on='date', how='outer')

        df4 = pd.merge(df3, df_unemployment_rate_pivoted, on='date', how='outer')

        df5 = pd.merge(df4, df_inactives_pivoted, on='date', how='outer')

        df6 = pd.merge(df5, df_industry_production_prices, on='date', how='outer')

        df7 = pd.merge(df6, df_gasoline_car_price, on='date', how='inner')

        df8 = pd.merge(df7, df_diesel_price, on='date', how='inner')

        df9 = pd.merge(df8, df_GPL_price, on='date', how='inner')

        df10 = pd.merge(df9, df_diesel_heatining_price, on='date', how='inner')

        final_df = pd.merge(df10, df_companies_trust, on='date', how='inner')

        final_df.rename(columns = {'workforce (Y15-64)': 'workforce', 
                                'inactives (Y15-64)': 'inactives',
                                'youth employment rate (Y15-34)': 'youth_employment_rate', 
                                'employment rate (Y15-64)':'employment_rate',  
                                'unemployment rate (Y15-64)': 'unemployment_rate', 
                                'activity rate (Y15-64)': 'activity_rate' }, inplace =True)

        final_df['date'] = pd.to_datetime(final_df['date'])
        final_df['month'] = final_df['date'].dt.month

        final_df = final_df[['date', 
                             'month',
                            'workforce', 
                            'inactives',
                            'employment_rate', 
                            'activity_rate', 
                            'unemployment_rate', 
                            'industry_production_prices', 
                            'companies_trust', 
                            'gasoline_price', 
                            'diesel_price', 
                            'GPL_price', 
                            'diesel_heatining_price']]

        final_df[['employment_rate', 'activity_rate', 'unemployment_rate']] = final_df[['employment_rate', 
                                                                                        'activity_rate', 'unemployment_rate']].round(2)

        return final_df




def cleaning(df):

    df["date"] = pd.to_datetime(df["date"], format='ISO8601' )
    df = df[df['date']<='2026-01-01'] # I filter only where I have rows without null values

    meadian_comp_trust= df['companies_trust'].median()
    meadian_ind_price = df['industry_production_prices'].median()
    df['companies_trust'] = df['companies_trust'].fillna(meadian_comp_trust )
    df['industry_production_prices'] = df['industry_production_prices'].fillna(meadian_ind_price )

    list_selected = ['date', 
                        'month',
                        'workforce', 
                        'inactives', 
                        'employment_rate',
                        'activity_rate', 
                        'unemployment_rate',
                        'industry_production_prices', 
                        'companies_trust', 
                        'gasoline_price',
                        'diesel_price', 
                        'GPL_price', 
                        'diesel_heatining_price']

    df = df[list_selected]

    df[[  'workforce', 'inactives']] = df[[ 'workforce', 'inactives']]*1000

    df[[  'workforce', 'inactives']] = df[[  'workforce', 'inactives']].astype(int)

    return df



def check_stationarity(df):
    non_stationary_cols = [
                    'workforce',
                    'inactives',
                    'employment_rate',
                    'activity_rate',
                    'industry_production_prices',
                    'gasoline_price',
                    'diesel_price',
                    'diesel_heatining_price',
                    'unemployment_rate'
                    ]

    for col in non_stationary_cols:
        # print(col)
        df[f'{col}_diff'] = df[col] - df[col].shift( 1) # qui facciamo subito la differnziazione

    return df



def select_cols_and_rolling(df):
    columns_selected = ['date',   
                        'month',
                        'workforce_diff', 
                        'inactives_diff',  
                        'employment_rate_diff', 
                        'activity_rate_diff', 
                        'industry_production_prices_diff',
                        'companies_trust',  
                        'gasoline_price_diff', 
                        'diesel_price_diff',  
                        'GPL_price',  
                        'diesel_heatining_price_diff', 
                        'unemployment_rate', 
                        'unemployment_rate_diff'
                        ]

    df = df[columns_selected]

    df['rolling_mean_6m'] = df['unemployment_rate_diff'].rolling(window=6).mean()
    df['rolling_mean_12m'] = df['unemployment_rate_diff'].rolling(window=12).mean()

    return df


def generate_PCA(df):
    df_merged_detrended = df.iloc[13:].copy() # questo serve a togliere le prime 13 righe che hanno dei valori NUll

    X_labour_market = df_merged_detrended[[  'workforce_diff', 'inactives_diff', 'activity_rate_diff']]

    X_price_energy = df_merged_detrended[[ 'gasoline_price_diff', 'diesel_price_diff', 'GPL_price', 'diesel_heatining_price_diff']]

    scaler = StandardScaler()
    X_labour_market_scaled = scaler.fit_transform(X_labour_market)
    X_price_energy_scaled = scaler.fit_transform(X_price_energy)

    pca = PCA(n_components=2)
    df_merged_detrended[['PCA_labour_market_1', 'PCA_labour_market_2']] = pca.fit_transform(X_labour_market_scaled)
    df_merged_detrended[['PCA_price_energy_1', 'PCA_price_energy_2']] = pca.fit_transform(X_price_energy_scaled)

    return df_merged_detrended


def preprocessing(df):
    df_preprocessed = cleaning(df)
    print('\n--cleaned--')
    df_stationary = check_stationarity(df_preprocessed)
    print('\n--stationary--')
    df_rolled = select_cols_and_rolling(df_stationary)
    print('\n--rollings--')
    df_pca = generate_PCA(df_rolled)
    print('\n--PCA--')    

    return df_pca



def predicting_forecast_dataset(df,selected_cols ):
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.index.name = None    

    df_fcst = pd.DataFrame()

    for feature in selected_cols:

        h = 12
        model = ARIMA( endog=df[feature]
                    , order=(0, 0, 13))

        res = model.fit()
        df_fcst[f'{feature}_fcst'] = res.forecast(steps = h)

    df_fcst['month'] = df_fcst.index.month    

    df_fcst = df_fcst[['month', 'employment_rate_diff_fcst', 'industry_production_prices_diff_fcst',
                        'companies_trust_fcst', 'rolling_mean_6m_fcst', 'rolling_mean_12m_fcst',
                        'PCA_labour_market_1_fcst', 'PCA_labour_market_2_fcst',
                        'PCA_price_energy_1_fcst', 'PCA_price_energy_2_fcst']]

    return df_fcst


if __name__ == "__main__":
    start_year_month = '2003-01'
    end_year_month = '2026-07'
    df = retrieving_data( start_year_month, end_year_month)
    path = Path(__file__).resolve().parent.parent / "datasets"
    path.mkdir(exist_ok=True)
    df_to_csv = df.copy()
    df_to_csv.to_csv(path / "dataset_raw.csv", index= False)
   
    df_training = preprocessing(df)
    df_training.to_csv(path / "dataset_preprocessed.csv", index = False)

    print(f'the variables are: {exogenous_variables}')
    df_input_forecast = predicting_forecast_dataset(df_training,exogenous_variables )

    df_input_forecast.to_csv(path / "dataset_input_forecast.csv", index = False)
    print("Finished!")




