"""
Preprocessing for Convertible Bond (CB) Terms and Bond Rate

Includes:
- Merging bond yield and credit data
- Constructing CB coupon term dictionaries
- Utility functions for bond pricing and implied volatility estimation
"""

import os
import pickle
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from tqdm import tqdm
from scipy.stats import norm
from scipy.optimize import brentq
from pricing_methods.bs import bs_call
from . import get_config


def get_bond_rate_data(read_from_file = True) -> pd.DataFrame:
    """Load or build corporate bond rates with rating and term."""
    # Load pricing parameters from config
    _config = get_config()

    files = _config['files']

    output_path = Path(files['bond_rate_output'])
    if read_from_file and output_path.exists():
        return pd.read_pickle(output_path)

    bond_yield = pd.read_csv(files['bond_yield'])
    bond_index = pd.read_csv(files['bond_index'])

    # Match bond types
    corp_bonds = bond_index[bond_index['Type'] == 'Corporate']
    treasury_bonds = bond_index[bond_index['Type'] == 'Treasury']

    # Corporate bond rates
    df_corp = pd.merge(corp_bonds, bond_yield, on='WindID', how='left')
    df_corp.rename(columns={"TradingDay": "TRADE_DT", "Close": "BOND_RATE"}, inplace=True)
    df_corp = df_corp[['TRADE_DT', 'WindID', 'CREDITRATING', 'Period', 'BOND_RATE']]
    df_corp.to_pickle(output_path)

    # Treasury bond rates (optional for reference)
    df_treasury = pd.merge(treasury_bonds, bond_yield, on='WindID', how='left')
    df_treasury.rename(columns={"TradingDay": "TRADE_DT", "Close": "BOND_RATE"}, inplace=True)
    df_treasury = df_treasury[['TRADE_DT', 'WindID', 'Period', 'BOND_RATE']]
    df_treasury.to_pickle(files['bond_rate_cn_output'])

    print("[SUCCESS] Bond rate data saved.")
    return df_corp


def load_cb_terms_dict(read_from_file = True) -> dict:
    """Load or build convertible bond coupon term dictionary."""
    # Load pricing parameters from config
    _config = get_config()

    files = _config['files']

    output_path = Path(files['cb_terms_pickle'])
    if read_from_file and output_path.exists():
        with open(output_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv(files['cb_coupon'])
    df = df[['wind_code', 'cash_flows_date', 'cash_flows_per_cny100_par', 'cf_type', 'coupon_rate']]
    df.sort_values(by=['wind_code', 'cash_flows_date'], inplace=True)

    cb_terms = {}
    for code, group in tqdm(df.groupby('wind_code'), desc="Building CB term dict"):
        cb_terms[code.split('.')[0]] = {
            'code': code,
            'Coupon_rate': group['coupon_rate'].tolist(),
            'Coupon': group['cash_flows_per_cny100_par'].tolist(),
            'date': group['cash_flows_date'].tolist(),
            'numYears': len(group)
        }

    with open(output_path, 'wb') as f:
        pickle.dump(cb_terms, f)
    print("[SUCCESS] Convertible bond term dictionary saved.")
    return cb_terms


def calculate_bond_price(term: dict, t: int, r: float) -> float:
    """
    Calculate the pure bond value of a CB based on future coupon flows.
    
    Args:
        term (dict): Bond term info
        t (int): Current date (e.g. 20230101)
        r (float): Annual interest rate (e.g. 0.04)

    Returns:
        float: Discounted bond value
    """
    idx_future = [i for i, d in enumerate(term['date']) if t < int(d.replace("-", ""))]
    t_dt = dt.datetime.strptime(str(t), '%Y%m%d')
    tau = np.array([(dt.datetime.strptime(term['date'][i], '%Y-%m-%d') - t_dt).days / 365.0 for i in idx_future])
    coupons = np.array([term['Coupon'][i] for i in idx_future])
    return float(np.sum(coupons * np.exp(-r * tau)))


def calculate_call_option_implied_volatility(price: float, s: float, k: float, tau: float, r: float,
                                             error_bound: float = 1e-4) -> float:
    """Estimate implied volatility from a call option price using bisection."""
    a, b = 0.01, 2.0
    va, _ = bs_call(s, k, tau, a, r)
    vb, _ = bs_call(s, k, tau, b, r)
    if (va - price) * (vb - price) > 0:
        return a if abs(va - price) < abs(vb - price) else b

    while True:
        sigma = (a + b) / 2.0
        v, _ = bs_call(s, k, tau, sigma, r)
        if abs(v - price) <= error_bound:
            return sigma
        if (v - price) * (va - price) > 0:
            a = sigma
        else:
            b = sigma


def scipy_implied_volatility(s: float, k: float, r: float, tau: float, market_price: float) -> float:
    """Estimate implied volatility using Brent's method."""

    def f(sigma):
        d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return s * norm.cdf(d1) - k * np.exp(-r * tau) * norm.cdf(d2) - market_price

    return brentq(f, 0.01, 1.0)


if __name__ == "__main__":
    print("[INFO] Running CB term and bond rate preprocessing.")

    bond_rate_df = get_bond_rate_data(read_from_file=False)
    print(f"[INFO] Loaded {len(bond_rate_df)} bond rate records.")

    cb_terms = load_cb_terms_dict(read_from_file=False)
    print(f"[INFO] Loaded {len(cb_terms)} convertible bond term records.")

    sample_code = list(cb_terms.keys())[0]
    sample_term = cb_terms[sample_code]
    price = calculate_bond_price(sample_term, t=20221231, r=0.035)
    print(f"[Example] Sample bond {sample_code} pure bond price: {price:.2f}")
