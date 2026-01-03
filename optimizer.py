import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

print("Starting ETF Portfolio Optimizer...\n")

# Load price data
prices = pd.read_csv("genericTestData/prices.csv", index_col=0, parse_dates=True)
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



# Load holdings structure
hold = pd.read_csv("genericTestData/holdings.csv")

countries = hold["Country"].unique().tolist()
C = len(countries)
print(f"Loaded {C} countries")

A = np.zeros((C, N))
TER = np.zeros(N)

for i, etf in enumerate(etfs):
    df = hold[hold["ETF"] == etf]
    TER[i] = df["TER"].iloc[0]
    
    for j, country in enumerate(countries):
        weight = df[df["Country"] == country]["Weight"].sum()
        A[j, i] = weight

# Optimization parameters
alpha = 0.5  # variance penalty
beta = 0.5   # correlation penalty
gamma = 3.0  # return reward
delta = 0.7  # TER penalty
lam = 0.1    # L2 concentration penalty

E_min = np.zeros(C)
E_max = np.ones(C) * 0.40
w_max = np.ones(N) * 0.30

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

for c in range(C):
    exposure = quicksum(A[c, i] * w[i] for i in range(N))
    m.addConstr(exposure >= E_min[c], f"country_min_{countries[c]}")
    m.addConstr(exposure <= E_max[c], f"country_max_{countries[c]}")
    
    # testing specific country constraints
   # if countries[c] == "Germany":
    #    m.addConstr(exposure >= 0.15, f"country_min_Germany_15pct")
    

# Solve
print("Solving...")
m.optimize()

# Print results
if m.status == GRB.OPTIMAL:
    print("\n" + "="*60)
    print("OPTIMAL SOLUTION FOUND")
    print("="*60)
    
    print("\nPortfolio Weights:")
    for i in range(N):
        print(f"  {etfs[i]:15s}: {w[i].X:7.4f} ({w[i].X*100:5.2f}%)")

    print("\nCountry Exposures:")
    for c in range(C):
        exposure_val = sum(A[c, i] * w[i].X for i in range(N))
        print(f"  {countries[c]:15s}: {exposure_val:7.4f} ({exposure_val*100:5.2f}%)")

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
else:
    print(f"\nOptimization failed with status: {m.status}")
