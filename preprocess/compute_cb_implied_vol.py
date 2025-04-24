"""
Compute implied volatility and bond price for convertible bonds.

This module estimates the implied volatility of the embedded call option
within a convertible bond by separating the bond component and using
Black-Scholes pricing with binary search.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from scipy.stats import norm
from . import get_config
from pricing_methods.bs import bs_call


def compute_cb_implied_vol(CBterms: dict) -> pd.DataFrame:
    """
    Load merged CB-stock dataset and compute implied volatility + bond value.

    Args:
        CBterms (dict): The coupon structure for each CB

    Returns:
        pd.DataFrame: Contains ['TRADE_DT', 'TRADE_CODE', 'tau', 'BondPrice', 'ImpliedVolatility']
    """
    # Load pricing parameters from config
    _config = get_config()

    data_path = _config['data_path']
    input_file = os.path.join(data_path, 'cb_with_stock_all.pkl')
    output_file = os.path.join(data_path, 'cb_implied_volatility.pkl')

    if os.path.exists(output_file):
        print("[INFO] Loaded existing implied volatility results.")
        return pd.read_pickle(output_file)

    print("[INFO] Calculating implied volatilities and bond prices.")

    df = pd.read_pickle(input_file)
    df['tau'] = ((df['Maturity'] - pd.to_datetime(df['TRADE_DT'].astype(str), format='%Y%m%d')) / np.timedelta64(1, 'D') + 1) / 365.0
    df['BondPrice'] = 0.0
    df['ImpliedVolatility'] = 0.0

    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            code = str(row['TRADE_CODE'])
            term = CBterms[code]
            t = int(row['TRADE_DT'])
            r = row['BOND_RATE'] / 100.0
            S0 = row['S_DQ_CLOSE']
            conv_price = row['CLAUSE_CONVERSION2_SWAPSHAREPRICE']
            tau = row['tau']

            # Compute bond value
            bond_val = compute_pure_bond_price(term, t, r)
            df.at[i, 'BondPrice'] = bond_val

            # Option price = market price - bond value (normalized)
            option_val = max((row['CLOSE_DP'] - bond_val) * conv_price / 100.0, 0)

            # Compute implied volatility using binary search
            df.at[i, 'ImpliedVolatility'] = implied_volatility_bs_call(
                    option_val, S0, term['Coupon'][-1] * conv_price / 100.0, tau, r
            )

        except Exception as e:
            print(f"[ERROR] Failed on row {i}: {e}")
            continue

    df_out = df[['TRADE_DT', 'TRADE_CODE', 'tau', 'BondPrice', 'ImpliedVolatility']].copy()
    df_out.to_pickle(output_file)
    print("[SUCCESS] Implied volatility data successfully saved.")
    return df_out


def compute_pure_bond_price(term: dict, t: int, r: float) -> float:
    """
    Compute the discounted value of future cash flows (pure bond price).

    Args:
        term (dict): Coupon structure
        t (int): Trade date as int (e.g., 20240101)
        r (float): Annual risk-free rate

    Returns:
        float: Present value of bond component
    """
    t_dt = datetime.strptime(str(t), '%Y%m%d')
    future_idx = [i for i, d in enumerate(term['date']) if t < int(d.replace("-", ""))]
    tau = np.array([(datetime.strptime(term['date'][i], '%Y-%m-%d') - t_dt).days / 365.0 for i in future_idx])
    coupons = np.array([term['Coupon'][i] for i in future_idx])
    return np.sum(coupons * np.exp(-r * tau))


def implied_volatility_bs_call(price, S, K, tau, r, tol = 1e-4, max_iter = 100) -> float:
    """
    Estimate implied volatility using binary search on Black-Scholes formula.

    Args:
        price (float): Observed option price
        S (float): Spot price of the underlying
        K (float): Strike price
        tau (float): Time to maturity (in years)
        r (float): Risk-free rate
        tol (float): Tolerance
        max_iter (int): Max iterations

    Returns:
        float: Estimated implied volatility
    """
    low, high = 0.01, 2.0
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        model_price, _ = bs_call(S, K, tau, mid, r)
        if abs(model_price - price) < tol:
            return mid
        if model_price > price:
            high = mid
        else:
            low = mid
    return mid


if __name__ == '__main__':
    from generate_cb_terms import load_or_create_cb_terms

    cb_terms = load_or_create_cb_terms()
    df_vol = compute_cb_implied_vol(cb_terms)
    print("[INFO] Sample output:")
    print(df_vol.head())
