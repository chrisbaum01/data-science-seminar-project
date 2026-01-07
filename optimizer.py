import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

print("Starting ETF Portfolio Optimizer...\n")

# Load price data
prices = pd.read_csv("generic_test_data/prices.csv", index_col=0, parse_dates=True)
prices = prices.sort_index()

etfs = prices.columns.tolist()
N = len(etfs)
print(f"Loaded {N} ETFs with {len(prices)} price observations")

# Compute returns and risk metrics
returns = np.log(prices / prices.shift(1)).dropna()
mu = returns.mean().values * 252
Sigma = returns.cov().values * 252
R = returns.corr().values
C_off = R - np.eye(N)

#Computed annualized returns and covariance matrix
print("\nAnnualized Expected Returns:")
for i in range(N):
    print(f"  {etfs[i]:15s}: {mu[i]:.4f} ({mu[i]*100:5.2f}%)")

print("\nAnnualized Covariance Matrix:")
for i in range(N):
    row = "  ".join(f"{Sigma[i, j]:.6f}" for j in range(N))
    print(f"  {row}")



# Load holdings country structure
hold_country = pd.read_csv("generic_test_data/holdings_country.csv")

countries = hold_country["Country"].unique().tolist()
C = len(countries)
print(f"Loaded {C} countries")

A = np.zeros((C, N))
TER = np.zeros(N)

for i, etf in enumerate(etfs):
    df = hold_country[hold_country["ETF"] == etf]
    TER[i] = df["TER"].iloc[0]
    
    for j, country in enumerate(countries):
        weight_c = df[df["Country"] == country]["Weight"].sum()
        A[j, i] = weight_c

# load holdings industry structure
hold_industry = pd.read_csv("generic_test_data/holdings_industry.csv")

industries = hold_industry["Industry"].unique().tolist()
I = len(industries)
print(f"Loaded {I} industries")
B = np.zeros((I, N))
for i, etf in enumerate(etfs):
    df = hold_industry[hold_industry["ETF"] == etf]
    
    for j, industry in enumerate(industries):
        weight_i = df[df["Industry"] == industry]["Weight"].sum()
        B[j, i] = weight_i


# Optimization parameters - play with these to see effects
alpha = 1.0  # variance penalty
beta = 0.5   # correlation penalty
gamma = 2.5  # return reward
delta = 0.2  # TER penalty
lam = 0.1    # L2 concentration penalty
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
    
    # Count active ETFs
    active_etfs = sum(1 for i in range(N) if w[i].X > 1e-6)
    print(f"\nNumber of ETFs in portfolio: {active_etfs} (max allowed: {max_etfs})")
    
    print("\nPortfolio Weights:")
    for i in range(N):
        if w[i].X > 1e-6:  # Only show non-zero weights
            print(f"  {etfs[i]:15s}: {w[i].X:7.4f} ({w[i].X*100:5.2f}%)")

    print("\nCountry Exposures:")
    for c in range(C):
        exposure_val = sum(A[c, i] * w[i].X for i in range(N))
        print(f"  {countries[c]:15s}: {exposure_val:7.4f} ({exposure_val*100:5.2f}%)")

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

        country_labels = countries
        country_sizes = [sum(A[c, i] * w[i].X for i in range(N)) for c in range(C)]
        
        # Filter out very small exposures for cleaner display
        threshold = 0.01
        filtered_data = [(label, size) for label, size in zip(country_labels, country_sizes) if size > threshold]
        if filtered_data:
            country_labels_filtered, country_sizes_filtered = zip(*filtered_data)
        else:
            country_labels_filtered, country_sizes_filtered = country_labels, country_sizes
        
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

        industry_labels = industries
        industry_sizes = [sum(B[ind, i] * w[i].X for i in range(N)) for ind in range(I)]
        
        # Filter out very small exposures for cleaner display
        threshold = 0.01
        filtered_data = [(label, size) for label, size in zip(industry_labels, industry_sizes) if size > threshold]
        if filtered_data:
            industry_labels_filtered, industry_sizes_filtered = zip(*filtered_data)
        else:
            industry_labels_filtered, industry_sizes_filtered = industry_labels, industry_sizes
        
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
