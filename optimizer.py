import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

print("Starting ETF Portfolio Optimizer...\n")


prices_long = pd.read_csv("webscrap_and_data/prices_6y.csv")

prices = prices_long.pivot(index='Date', columns='Ticker', values='value')
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()


print(f"Loaded {len(prices.columns)} ETFs with {len(prices)} price observations")
missing_pct = prices.isna().sum() / len(prices) * 100
print(f"ETFs with >50% missing data: {(missing_pct > 50).sum()}")

# Drop ETFs with more than 50% missing data
prices = prices.loc[:, missing_pct <= 50]
# Forward fill remaining missing values (assume price unchanged on missing days)
prices = prices.fillna(method='ffill')
prices = prices.dropna(axis=1)

etfs = prices.columns.tolist()
N = len(etfs)
print(f"After data cleaning: {N} ETFs with {len(prices)} price observations")

# Compute returns and risk metrics
returns = np.log(prices / prices.shift(1)).dropna()
mu = returns.mean().values * 252
Sigma = returns.cov().values * 252
R = returns.corr().values
C_off = R - np.eye(N)

#Computed annualized returns and covariance matrix - only print mu and variance for EUNL.DE

if 'EUNL.DE' in etfs:
    idx = etfs.index('EUNL.DE')
    print("\nAnnualized Expected Returns:")
    print(f"  {'EUNL.DE':15s}: {mu[idx]:.4f} ({mu[idx]*100:5.2f}%)")
    variance = Sigma[idx, idx]
    print(f"\nVariance of EUNL.DE: {variance:.6f} ({np.sqrt(variance)*100:.2f}% volatility)")
else:
    print("\nEUNL.DE not found in ETF list")


def strip_exchange_suffix(ticker):
    """Remove exchange suffix like .DE, .L, .SW, .AS from ticker"""
    for suffix in ['.DE', '.L', '.SW', '.AS']:
        if ticker.endswith(suffix):
            return ticker[:-len(suffix)]
    return ticker


hold_country = pd.read_csv("webscrap_and_data/master_location_table.csv")


hold_country = hold_country[~hold_country["Location"].isin(["Europäische Union", "--"])]

countries = hold_country["Location"].unique().tolist()
C = len(countries)
print(f"Loaded {C} countries")

A = np.zeros((C, N))
TER = np.zeros(N)

for i, etf in enumerate(etfs):
    # Strip exchange suffix to match with holdings data
    etf_base = strip_exchange_suffix(etf)
    df = hold_country[hold_country["Ticker"] == etf_base]
    if len(df) > 0:
        TER[i] = df["TER"].iloc[0]
    
    for j, country in enumerate(countries):
        weight_c = df[df["Location"] == country]["Weight"].sum()
        A[j, i] = weight_c


hold_industry = pd.read_csv("webscrap_and_data/master_industry_table.csv")

industries = hold_industry["Sector"].unique().tolist()
I = len(industries)
print(f"Loaded {I} industries")
B = np.zeros((I, N))
for i, etf in enumerate(etfs):
    # Strip exchange suffix to match with holdings data
    etf_base = strip_exchange_suffix(etf)
    df = hold_industry[hold_industry["Ticker"] == etf_base]
    
    for j, industry in enumerate(industries):
        weight_i = df[df["Sector"] == industry]["Weight"].sum()
        B[j, i] = weight_i


# Optimization parameters

# Low-risk profile
'''
alpha = 2.0   # strong variance penalty
beta  = 4.0   # strong correlation penalty
gamma = 0.8   # low return reward
delta = 0.3   # moderate TER penalty
lam   = 0.5   # strong concentration penalty
profile_name = "Low-risk profile selected"

# Medium-risk profile
alpha = 1.0   # balanced variance penalty
beta  = 2.5   # balanced correlation penalty
gamma = 2.0   # moderate return reward
delta = 0.2   # moderate TER penalty
lam   = 0.2   # moderate concentration penalty
profile_name = "Medium-risk profile selected"
'''
# High-risk profile
alpha = 0.5   # weak variance penalty
beta  = 1.0   # weak correlation penalty
gamma = 4.0   # strong return reward
delta = 0.1   # low TER penalty
lam   = 0.05  # weak concentration penalty
profile_name = "High-risk profile selected"

max_etfs = 10  



E_min = np.zeros(C)
E_max = np.ones(C) * 0.30 
I_min = np.zeros(I)
I_max = np.ones(I) * 0.30 
w_max = np.ones(N) * 0.50 

# Build optimization model
print("\nBuilding optimization model...")
m = Model("ETF_Portfolio_Optimizer")

w = m.addVars(N, lb=0.0, ub=w_max.tolist(), name="w")

var_term = quicksum(w[i] * w[j] * Sigma[i, j] for i in range(N) for j in range(N))
corr_term = quicksum(w[i] * w[j] * C_off[i, j] for i in range(N) for j in range(N))
ret_term = quicksum(mu[i] * w[i] for i in range(N))
ter_term = quicksum(TER[i] * w[i] for i in range(N))
l2_term = quicksum(w[i] * w[i] for i in range(N))


m.setObjective(
    alpha * var_term +
    beta * corr_term -
    gamma * ret_term +
    delta * ter_term +
    lam * l2_term,
    GRB.MINIMIZE
)

# Add constraints

m.addConstr(quicksum(w[i] for i in range(N)) == 1, "budget")


for i in range(N):
    m.addConstr(w[i] >= 0.0, f"weight_min_{etfs[i]}")
    m.addConstr(w[i] <= w_max[i], f"weight_max_{etfs[i]}")


for c in range(C):
    exposure = quicksum(A[c, i] * w[i] for i in range(N))
    m.addConstr(exposure >= E_min[c], f"country_min_{countries[c]}")
    m.addConstr(exposure <= E_max[c], f"country_max_{countries[c]}")
    
    # testing specific country constraints
   # if countries[c] == "USA":
    #    m.addConstr(exposure <= 0.15, f"country_max_USA_15pct")



for ind in range(I):
    exposure = quicksum(B[ind, i] * w[i] for i in range(N))
    m.addConstr(exposure >= I_min[ind], f"industry_min_{industries[ind]}")
    m.addConstr(exposure <= I_max[ind], f"industry_max_{industries[ind]}")


y = m.addVars(N, vtype=GRB.BINARY, name="y")
M = 1.0  # big-M value (should be >= max possible weight)
for i in range(N):
    m.addConstr(w[i] <= M * y[i], f"bigM_{etfs[i]}")
m.addConstr(quicksum(y[i] for i in range(N)) <= max_etfs, "max_etfs_constraint")
    

# Solve
print("Solving...")
m.optimize()

# Print results
if m.status == GRB.OPTIMAL:
    print("\n" + "="*60)
    print("OPTIMAL SOLUTION FOUND")
    print("="*60)
    print(f"  Risk profile:       {profile_name}")

    # Count active ETFs
    active_etfs = sum(1 for i in range(N) if w[i].X > 1e-6)
    print(f"\nNumber of ETFs in portfolio: {active_etfs} (max allowed: {max_etfs})")
    
    print("\nPortfolio Weights:")
    for i in range(N):
        if w[i].X > 1e-6:  # Only show non-zero weights
            print(f"  {etfs[i]:15s}: {w[i].X:7.4f} ({w[i].X*100:5.2f}%)")

    print("\nCountry Exposures:")
    # only top 10 countries by exposure
    country_exposures = [(countries[c], sum(A[c, i] * w[i].X for i in range(N))) for c in range(C)]
    country_exposures_sorted = sorted(country_exposures, key=lambda x: x[1], reverse=True)[:10]
    for country, exposure_val in country_exposures_sorted:
        print(f"  {country:15s}: {exposure_val:7.4f} ({exposure_val*100:5.2f}%)")

    print("\nIndustry Exposures:")
    for ind in range(I):
        exposure_val = sum(B[ind, i] * w[i].X for i in range(N))
        print(f"  {industries[ind]:15s}: {exposure_val:7.4f} ({exposure_val*100:5.2f}%)")

    # Portfolio statistics
    var_val = sum(w[i].X * w[j].X * Sigma[i, j] for i in range(N) for j in range(N))
    ret_val = sum(mu[i] * w[i].X for i in range(N))
    ter_val = sum(TER[i] * w[i].X for i in range(N))
  

    
    print("\nPortfolio Statistics:")
    
    print(f"  Expected return:    {ret_val:.4f} ({ret_val*100:.2f}%)")
    print(f"  Volatility:         {np.sqrt(var_val):.4f} ({np.sqrt(var_val)*100:.2f}%)")
    print(f"  Sharpe ratio:       {ret_val / np.sqrt(var_val):.4f}")
    print(f"  Weighted avg TER:   {ter_val:.4f} ({ter_val*100:.2f}%)")
    print(f"  Objective value:    {m.objVal:.6f}")
