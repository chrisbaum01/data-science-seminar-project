import numpy as np
import pandas as pd

np.random.seed(42)

# ===============================
# Parameters
# ===============================

n_years = 5
trading_days = 252 * n_years
dates = pd.bdate_range("2019-01-01", periods=trading_days)

etfs = [
    "ETF_US",
    "ETF_EU",
    "ETF_EM",
    "ETF_ASIA",
    "ETF_BOND",
    "ETF_COMMOD"
]

N = len(etfs)

# Annualized expected returns
mu_annual = np.array([0.08, 0.07, 0.10, 0.085, 0.03, 0.05])

# Annualized volatilities
vol_annual = np.array([0.16, 0.15, 0.22, 0.18, 0.05, 0.20])

# Correlation matrix
corr = np.array([
    [1.00, 0.85, 0.70, 0.75, 0.20, 0.40],
    [0.85, 1.00, 0.65, 0.70, 0.25, 0.35],
    [0.70, 0.65, 1.00, 0.75, 0.15, 0.45],
    [0.75, 0.70, 0.75, 1.00, 0.20, 0.40],
    [0.20, 0.25, 0.15, 0.20, 1.00, 0.10],
    [0.40, 0.35, 0.45, 0.40, 0.10, 1.00],
])

# ===============================
# Covariance matrix
# ===============================

cov_annual = np.outer(vol_annual, vol_annual) * corr
cov_daily = cov_annual / 252
mu_daily = mu_annual / 252

# ===============================
# Simulate daily log returns
# ===============================

returns = np.random.multivariate_normal(
    mean=mu_daily,
    cov=cov_daily,
    size=trading_days
)

returns = pd.DataFrame(returns, index=dates, columns=etfs)

# ===============================
# Convert to price series
# ===============================

prices = 100 * np.exp(returns.cumsum())
prices.to_csv("prices.csv")

print("✔ prices.csv generated")

# ===============================
# Holdings / country exposure
# ===============================

holdings = [
    # ETF_US
    ("ETF_US", "USA", 0.85, 0.0007),
    ("ETF_US", "Canada", 0.15, 0.0007),

    # ETF_EU
    ("ETF_EU", "Germany", 0.30, 0.0012),
    ("ETF_EU", "France", 0.25, 0.0012),
    ("ETF_EU", "Italy", 0.25, 0.0012),
    ("ETF_EU", "Spain", 0.20, 0.0012),

    # ETF_EM
    ("ETF_EM", "China", 0.35, 0.0020),
    ("ETF_EM", "India", 0.30, 0.0020),
    ("ETF_EM", "Brazil", 0.20, 0.0020),
    ("ETF_EM", "Mexico", 0.15, 0.0020),

    # ETF_ASIA
    ("ETF_ASIA", "Japan", 0.40, 0.0015),
    ("ETF_ASIA", "South Korea", 0.25, 0.0015),
    ("ETF_ASIA", "Taiwan", 0.20, 0.0015),
    ("ETF_ASIA", "Singapore", 0.15, 0.0015),

    # ETF_BOND
    ("ETF_BOND", "USA", 0.50, 0.0004),
    ("ETF_BOND", "Germany", 0.30, 0.0004),
    ("ETF_BOND", "Japan", 0.20, 0.0004),

    # ETF_COMMOD
    ("ETF_COMMOD", "Global", 1.00, 0.0025),
]

holdings_df = pd.DataFrame(
    holdings, columns=["ETF", "Country", "Weight", "TER"]
)

holdings_df.to_csv("holdings.csv", index=False)

print("✔ holdings.csv generated")