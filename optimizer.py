import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

print("Starting ETF Portfolio Optimizer...\n")

# Load price data (long format with Date, Ticker, value columns)
prices_long = pd.read_csv("webscrap_and_data/prices_6y.csv")
# Convert from long format to wide format
prices = prices_long.pivot(index='Date', columns='Ticker', values='value')
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

# Handle missing values: forward fill then drop columns with too many NaNs
print(f"Loaded {len(prices.columns)} ETFs with {len(prices)} price observations")
missing_pct = prices.isna().sum() / len(prices) * 100
print(f"ETFs with >50% missing data: {(missing_pct > 50).sum()}")

# Drop ETFs with more than 50% missing data
prices = prices.loc[:, missing_pct <= 50]
# Forward fill remaining missing values (assume price unchanged on missing days)
prices = prices.fillna(method='ffill')
# Drop any rows with remaining NaNs (typically at the start)
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

#Computed annualized returns and covariance matrix
# only print mu and variance for EUNL.DE

if 'EUNL.DE' in etfs:
    idx = etfs.index('EUNL.DE')
    print("\nAnnualized Expected Returns:")
    print(f"  {'EUNL.DE':15s}: {mu[idx]:.4f} ({mu[idx]*100:5.2f}%)")
    variance = Sigma[idx, idx]
    print(f"\nVariance of EUNL.DE: {variance:.6f} ({np.sqrt(variance)*100:.2f}% volatility)")
else:
    print("\nEUNL.DE not found in ETF list")

'''
print("\nAnnualized Covariance Matrix:")
for i in range(N):
    row = "  ".join(f"{Sigma[i, j]:.6f}" for j in range(N))
    print(f"  {row}")
'''

# Helper function to strip exchange suffix from ticker
def strip_exchange_suffix(ticker):
    """Remove exchange suffix like .DE, .L, .SW, .AS from ticker"""
    for suffix in ['.DE', '.L', '.SW', '.AS']:
        if ticker.endswith(suffix):
            return ticker[:-len(suffix)]
    return ticker

# Load holdings country structure
hold_country = pd.read_csv("webscrap_and_data/master_location_table.csv")

# Filter out EU and empty locations
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

# load holdings industry structure
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

max_etfs = 10  # maximum number of ETFs in portfolio


# max exposure for country and industries + weight constraints
E_min = np.zeros(C)
E_max = np.ones(C) * 0.30 # max 30% exposure per country
I_min = np.zeros(I)
I_max = np.ones(I) * 0.30 # max 30% exposure per industry
w_max = np.ones(N) * 0.50 # max 50% weight per ETF

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
# etf weight sum to 1 constraint
m.addConstr(quicksum(w[i] for i in range(N)) == 1, "budget")

# individual etf weight constraints
for i in range(N):
    m.addConstr(w[i] >= 0.0, f"weight_min_{etfs[i]}")
    m.addConstr(w[i] <= w_max[i], f"weight_max_{etfs[i]}")

#country exposure constraints
for c in range(C):
    exposure = quicksum(A[c, i] * w[i] for i in range(N))
    m.addConstr(exposure >= E_min[c], f"country_min_{countries[c]}")
    m.addConstr(exposure <= E_max[c], f"country_max_{countries[c]}")
    
    # testing specific country constraints
   # if countries[c] == "USA":
    #    m.addConstr(exposure <= 0.15, f"country_max_USA_15pct")


# add industry exposure constraints 
for ind in range(I):
    exposure = quicksum(B[ind, i] * w[i] for i in range(N))
    m.addConstr(exposure >= I_min[ind], f"industry_min_{industries[ind]}")
    m.addConstr(exposure <= I_max[ind], f"industry_max_{industries[ind]}")

# add constraint that we want a max of max_etfs ETFs in the portfolio
# This requires introducing binary variables and a big-M constraint
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

'''
    # add visualization of the portfolio weights
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use TkAgg backend for displaying plots
        import matplotlib.pyplot as plt

        labels = [etfs[i] for i in range(N) if w[i].X > 1e-6]
        sizes = [w[i].X for i in range(N) if w[i].X > 1e-6]
        
        # Create explode effect for top holdings
        explode = [0.05 if size == max(sizes) else 0 for size in sizes]
        
        # Use a nice color palette
        colors = plt.cm.Dark2(range(len(sizes)))

        plt.figure(figsize=(12, 8))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                            startangle=140, colors=colors, explode=explode,
                                            textprops={'fontsize': 10, 'weight': 'bold'})
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
        
        plt.axis('equal')
        plt.title('ETF Portfolio Weights', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.show(block=True)
    except ImportError:
        print("matplotlib not installed, skipping portfolio weights visualization.")
    # add visualization of country exposures
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        # Compute exact country exposures (same as printed)
        country_exposures = [(countries[c], sum(A[c, i] * w[i].X for i in range(N))) for c in range(C)]
        
        # Separate large and small exposures
        threshold = 0.01
        large_exposures = [(label, size) for label, size in country_exposures if size > threshold]
        small_exposures = [(label, size) for label, size in country_exposures if size <= threshold]
        
        # Add Rest category if there are small exposures
        if small_exposures:
            rest_sum = sum(size for _, size in small_exposures)
            large_exposures.append(('Rest', rest_sum))
        
        if large_exposures:
            country_labels_filtered, country_sizes_filtered = zip(*large_exposures)
        else:
            country_labels_filtered, country_sizes_filtered = ['No Data'], [1.0]
        
        # Create explode effect for top exposures
        explode = [0.05 if size == max(country_sizes_filtered) else 0 for size in country_sizes_filtered]
        
        # Use a nice color palette
        colors = plt.cm.Dark2(range(len(country_sizes_filtered)))

        plt.figure(figsize=(12, 8))
        wedges, texts, autotexts = plt.pie(country_sizes_filtered, labels=country_labels_filtered, 
                                            autopct='%1.1f%%', startangle=140, colors=colors,
                                            explode=explode, 
                                            textprops={'fontsize': 10, 'weight': 'bold'})
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
        
        plt.axis('equal')
        plt.title('Country Exposures in ETF Portfolio', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.show(block=True)
    except ImportError:
        print("matplotlib not installed, skipping country exposures visualization.")
    # add visualization of industry exposures
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        # Compute exact industry exposures (same as printed)
        industry_exposures = [(industries[ind], sum(B[ind, i] * w[i].X for i in range(N))) for ind in range(I)]
        
        # Separate large and small exposures
        threshold = 0.01
        large_exposures = [(label, size) for label, size in industry_exposures if size > threshold]
        small_exposures = [(label, size) for label, size in industry_exposures if size <= threshold]
        
        # Add Rest category if there are small exposures
        if small_exposures:
            rest_sum = sum(size for _, size in small_exposures)
            large_exposures.append(('Rest', rest_sum))
        
        if large_exposures:
            industry_labels_filtered, industry_sizes_filtered = zip(*large_exposures)
        else:
            industry_labels_filtered, industry_sizes_filtered = ['No Data'], [1.0]
        
        # Create explode effect for top exposures
        explode = [0.05 if size == max(industry_sizes_filtered) else 0 for size in industry_sizes_filtered]
        
        # Use a nice color palette
        colors = plt.cm.Dark2(range(len(industry_sizes_filtered)))

        plt.figure(figsize=(12, 8))
        wedges, texts, autotexts = plt.pie(industry_sizes_filtered, labels=industry_labels_filtered, 
                                            autopct='%1.1f%%', startangle=140, colors=colors,
                                            explode=explode, 
                                            textprops={'fontsize': 10, 'weight': 'bold'})
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
        
        plt.axis('equal')
        plt.title('Industry Exposures in ETF Portfolio', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        plt.show(block=True)
    except ImportError:
        print("matplotlib not installed, skipping industry exposures visualization.")

    # print plots


    
else:
    print(f"\nOptimization failed with status: {m.status}")
'''