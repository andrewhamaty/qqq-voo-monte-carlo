#%% IMPORTS
import datetime as dt

import numpy as np
import pandas as pd
from prettytable import PrettyTable
import yfinance as yf

from utils import *

#%% GET DATA
tickers = ['QQQ', 'VOO']
start_date = '2015-01-01' # let's get around 10 years worth of data
TODAY = dt.date.today()
data = yf.download(tickers, start=start_date, end=TODAY, auto_adjust=True)['Close']

#%% CALCULATE RETURNS AND OTHER STATS
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

#%% DEFINE THE SIMULATION PARAMS
trading_days = 252
n_simulations = 10000
n_days =trading_days * 10

#%% RUN SIMULATIONS
def simulate_paths(start_price: float, mu: float, sigma: float, days: int, n_simulations: int):

    simulations = np.zeros( (n_simulations, days) )
    simulations[:, 0] = start_price

    for t in range(1, days):
        random_returns = np.random.normal(loc=mu, scale=sigma, size=n_simulations)
        simulations[:, t] = simulations[:, t-1] * (1 + random_returns)

    return simulations

# run qqq and plot
qqq_simulation = simulate_paths(data['QQQ'].iloc[-1],
                                mean_returns['QQQ'],
                                returns['QQQ'].std(),
                                n_days,
                                n_simulations)

fig = plot_monte_carlo_paths(qqq_simulation, n_to_plot=1000, title='QQQ Monte Carlo Simulation')
fig.show()


# run voo and plot
voo_simulation = simulate_paths(data['VOO'].iloc[-1],
                                mean_returns['VOO'],
                                returns['VOO'].std(),
                                n_days,
                                n_simulations)

fig = plot_monte_carlo_paths(voo_simulation, n_to_plot=1000, title='VOO Monte Carlo Simulation')
fig.show()

""" 
### Interpretation of the Simulation Paths

* Both ETFs show exponential growth patterns over the simulation period
* QQQ exhibits higher volatility and greater upside potential, with some simulation paths reaching above $20k
* VOO shows more conservative growth with less extreme outcomes
"""

#%% COMPARE ENDING VALUES AND PLOT
qqq_ending = qqq_simulation[:, -1]
voo_ending = voo_simulation[:, -1]

fig = plot_kde_comparison(
    {'QQQ': qqq_ending, 'VOO': voo_ending},
    title='Simulated 10-Year Ending Values',
    xlabel='Portfolio Value')
fig.show()

"""
### Interpretations of Density Plot
* QQQ has a wider, faltter distribution extending further right, indicating higher potential returns, but also greater variability 
* QQQ's peak around 250μ at ~3k means portfolio values near $3,000 are the most likely outcome for QQQ
* VOO shows a more concentrated distribution with a sharper peak, suggesting more predictable but lower returns
* VOO's peak around 400μ at ~$3k means portfolio values near $3,000 are the most likely outcome for VOO
* QQQ's maximum density is lower than VOO's - the probability is more spread out
* Both ETF's peak around the same $3k value meaning that both ETFs have similar 'most-likely' outcomes
"""

#%% RISK ADJUSTED COMPARISON
def sharpe_ratio(returns, risk_free_rate=0.02):
    # Annualized
    return (returns.mean() * trading_days - risk_free_rate) / (returns.std() * np.sqrt(trading_days))

qqq_sharpe = sharpe_ratio(returns['QQQ'])
voo_sharpe = sharpe_ratio(returns['VOO'])

# display results
sharp_table = PrettyTable(['Ticker', 'Sharpe Ratio'])
sharp_table.add_row(['QQQ', np.round(qqq_sharpe, 3)])
sharp_table.add_row(['VOO', np.round(voo_sharpe, 3)])
print(sharp_table)

"""
# Interpretation of Sharpe Ratios
* The Sharpe Ratio measures risk-adjusted returns - aka how much extra return you receive for the volatility you endure
* QQQ is more efficient
    * Despite being more volatile, QQQ provides better risk adjusted returns
    * Investors are well compensated for taking on the additional risk
"""

# PROBABILITY OF QQQ UNDERPERFORMING VOO
prob_underperformance = np.mean(qqq_ending < voo_ending)
print(f"Probability QQQ underperforms VOO over 10 years: {prob_underperformance:.2%}")

"""
### Interpretation
* QQQ underperforms VOO 33.16% of the time
* Therefore, QQQ is the probabilistic winner
* This confirms the Sharpe Ratio we saw earlier
* While QQQ is riskier, you have twice the chance of coming out ahead when you invest in QQQ instead of VOO
* However, 33% is not negligible - whether or not this phases you depends on your risk tolerance

"""
# %%
