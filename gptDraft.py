import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum

# ===============================
# 1. Load data
# ===============================

# Price history (e.g. 5 years)
prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)
prices = prices.sort_index()

# ETF list
etfs = prices.columns.tolist()
N = len(etfs)

# Returns (daily log returns)
returns = np.log(prices / prices.shift(1)).dropna()

# Expected returns (mean daily return -> annualized)
mu = returns.mean().values * 252   # vector length N

# Covariance matrix (annualized)
Sigma = returns.cov().values * 252  # NxN

# Correlation matrix
R = returns.corr().values

# Off-diagonal correlation matrix C_off = R - I
C_off = R - np.eye(N)

# ===============================
# 2. Load holdings structure (country exposures + TER)
# ===============================

hold = pd.read_csv("holdings.csv")

# unique countries
countries = hold["Country"].unique().tolist()
C = len(countries)

# exposure matrix A (C x N)
A = np.zeros((C, N))

# TER vector
TER = np.zeros(N)

# Fill A and TER
for i, etf in enumerate(etfs):
    df = hold[hold["ETF"] == etf]
    TER[i] = df["TER"].iloc[0]  # assume same TER for all rows of that ETF
    
    for j, country in enumerate(countries):
        weight = df[df["Country"] == country]["Weight"].sum()
        A[j, i] = weight

# ===============================
# 3. Optimization parameters
# ===============================

# Hyperparameters
alpha = 1.0     # variance penalty
beta = 0.3      # correlation penalty
gamma = 0.5     # return reward
delta = 1.0     # TER penalty
lam = 0.1       # L2 concentration penalty

# Country bounds (example: min 0%, max 40% per country)
E_min = np.zeros(C)
E_max = np.ones(C) * 0.40

# Maximum ETF weight (e.g. 30%)
w_max = np.ones(N) * 0.30

# ===============================
# 4. Build optimization model
# ===============================

m = Model("ETF_Portfolio_Optimizer")

# Decision variables: w_i
w = m.addVars(N, lb=0.0, ub=w_max.tolist(), name="w")

# Portfolio variance term: w^T Σ w
var_term = quicksum(w[i] * w[j] * Sigma[i, j] for i in range(N) for j in range(N))

# Correlation penalty: w^T C_off w
corr_term = quicksum(w[i] * w[j] * C_off[i, j] for i in range(N) for j in range(N))

# Return reward: μ^T w
ret_term = quicksum(mu[i] * w[i] for i in range(N))

# TER penalty: TER^T w
ter_term = quicksum(TER[i] * w[i] for i in range(N))

# L2 penalty: sum w_i^2
l2_term = quicksum(w[i] * w[i] for i in range(N))

# Objective function
m.setObjective(
    alpha * var_term +
    beta * corr_term -
    gamma * ret_term +
    delta * ter_term +
    lam * l2_term,
    GRB.MINIMIZE
)

# ===============================
# 5. Constraints
# ===============================

# Full investment
m.addConstr(quicksum(w[i] for i in range(N)) == 1, "budget")

# Country exposure constraints
for c in range(C):
    exposure = quicksum(A[c, i] * w[i] for i in range(N))
    m.addConstr(exposure >= E_min[c], f"country_min_{countries[c]}")
    m.addConstr(exposure <= E_max[c], f"country_max_{countries[c]}")

# ===============================
# 6. Solve
# ===============================

m.optimize()

# ===============================
# 7. Print results
# ===============================

if m.status == GRB.OPTIMAL:
    print("\nOptimal Portfolio Weights:")
    for i in range(N):
        print(f"{etfs[i]:10s}: {w[i].X:.4f}")

    print("\nCountry Exposures:")
    for c in range(C):
        exposure_val = sum(A[c, i] * w[i].X for i in range(N))
        print(f"{countries[c]:10s}: {exposure_val:.4f}")

    print("\nObjective value:", m.objVal)
