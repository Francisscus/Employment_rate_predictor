from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "datasets"
MODEL_DIR = ROOT_DIR / "models"

# variables
exogenous_variables = ['month', 
                'employment_rate_diff', 
                'industry_production_prices_diff',
                'companies_trust',  
                'rolling_mean_6m',
                'rolling_mean_12m', 
                'PCA_labour_market_1',
                'PCA_labour_market_2', 
                'PCA_price_energy_1', 
                'PCA_price_energy_2'
                ]

target = 'unemployment_rate'
target_diff = 'unemployment_rate_diff'

test_split = -12

best_param = (0, 0, 0)
best_param_seasonal = (1, 0, 0, 12)





