"""
Black-Scholes Convertible Bond Pricing Module.
"""

import numpy as np
from scipy.stats import norm
import datetime as dt
from config_loader import load_config

# Load pricing parameters from config
_config = load_config()
pricing_config = _config["pricing"]

float_lower_bound = pricing_config["float_lower_bound"]
float_upper_bound = pricing_config["float_upper_bound"]
days_of_year = pricing_config["days_of_year"]


def bs_cb(s: float, term: dict, t: dt.datetime, implied_vol: float, r: float, bond_rate_dict: dict) -> (float, float):
    """
    Price a convertible bond using the Black-Scholes (BS) model.

    The convertible bond price is calculated as:
        Convertible Bond Price = Present Value of Straight Bond + Call Option Value * (100 / Conversion Price)

    The call option strike price K is calculated as:
        K = (Final coupon payment) * (Conversion Price) / 100

    The pricing is based on the closing stock price at time `t`. The remaining time to maturity (τ) is:
        τ = (Maturity Date - Current Date) in years

    Example:
        If the maturity date is 2019-12-25 and the current pricing date is 2019-12-24, then τ = 1.0.

    Note:
        The purpose of this pricing is to evaluate whether a convertible bond is over- or under-valued by the market.
        If the theoretical price is significantly higher than the market price, it may suggest a buying opportunity.

    Args:
        s (float): Closing price of the underlying stock on date `t`.
        term (dict): Dictionary containing convertible bond terms and key parameters.
        t (datetime): Current date for pricing.
        implied_vol (float): Implied volatility of the underlying stock.
        r (float): Risk-free interest rate (use np.log(1 + r) internally).
        bond_rate_dict (dict): Dictionary of bond rates, indexed by duration (in years), e.g. {1.0: 0.025, ...}

    Examples:
        A typical `term` dictionary for a convertible bond (e.g. '110030.SH'):

        {
            'ConvPrice': 7.24,  # Conversion price
            'Maturity': datetime.datetime(2019, 12, 25, 0, 0),  # Maturity date
            'ConvertStart': 4.5,  # Conversion window begins 4.5 years before maturity
            'Coupon': [0.6, 0.8, 1.0, 1.5, 106.0],  # Coupon payments; last includes principal
            'Coupon_rate': [0.6, 0.8, 1.0, 1.5, 2.0],
            'Coupon_date': ['2015-12-25', '2016-12-25', '2017-12-25', '2018-12-25', '2019-12-25'],
            'Coupon_date_dt': [
                datetime.datetime(2015, 12, 25, 0, 0),
                datetime.datetime(2016, 12, 25, 0, 0),
                datetime.datetime(2017, 12, 25, 0, 0),
                datetime.datetime(2018, 12, 25, 0, 0),
                datetime.datetime(2019, 12, 25, 0, 0)
            ],
            'Recall': [4.5, 20, 30, 9.412],  # Callable: [start year, trigger days, window, trigger price]
            'CallOptionTriggerProportion': 130.0,
            'Resell': [3.0, 30, 30, 5.068, 103],  # Puttable: [start year, trigger days, window, trigger price, put price]
            'RedeemTriggerProportion': 70.0
        }

    Returns:
        tuple:
            - float: Convertible bond price
            - float: Delta of the embedded call option
    """
    price_bond = calculate_bond(term, t, bond_rate_dict)
    K = term['Coupon'][-1] * term['ConvPrice'] / 100.0
    tau = (term['Maturity'] - t).days / days_of_year
    bs, delta = bs_call(s, K, tau, implied_vol, np.log(1 + r))
    price = price_bond + bs * 100 / term['ConvPrice']
    return price, delta


def calculate_bond(term: dict, t: dt.datetime, bond_rate_dict: dict) -> float:
    """
    Calculate present value of the pure bond component.

    Args:
        term (dict): CB term information
        t (datetime): Current pricing date
        bond_rate_dict (dict): Mapping: remaining years → corporate bond yield

    Returns:
        float: Present value of the bond component
    """
    tau = ((np.array(term['Coupon_date_dt']) - t) / dt.timedelta(days=1)) / days_of_year
    period = 0.5 * ((tau <= 0.5) * (tau > 0)) + np.ceil(tau) * (tau > 0.5)
    rate = list(map(bond_rate_dict.get, period))
    coupon = np.array(term['Coupon'])

    rate_clean = np.array([0 if r is None else r for r in rate], dtype=float)
    price = np.sum(np.exp(list(-tau * rate_clean / 100)) * coupon * (rate_clean > 0))
    return price


def bs_call(s: float, k: float, tau: float, implied_vol: float, r: float) -> (float, float):
    """
    Black-Scholes call option price and delta.

    Args:
        s (float): Stock price
        k (float): Strike price
        tau (float): Time to maturity (in years)
        implied_vol (float): Volatility
        r (float): Risk-free rate (log-form)

    Returns:
        (float, float): Call option value, delta
    """
    d1 = (np.log(s / k) + (r + 0.5 * implied_vol**2) * tau) / (implied_vol * np.sqrt(tau))
    d2 = d1 - implied_vol * np.sqrt(tau)
    cdf_d1, cdf_d2 = norm.cdf(d1), norm.cdf(d2)

    cdf_d1 = max(cdf_d1, float_lower_bound)
    cdf_d2 = max(cdf_d2, float_lower_bound)

    price = s * cdf_d1 - k * np.exp(-r * tau) * cdf_d2
    return price, cdf_d1


if __name__ == '__main__':
    # Sample convertible bond term data
    term_example = {
        'ConvPrice': 10.0,
        'Maturity': dt.datetime(2026, 12, 31),
        'Coupon': [1.0, 1.0, 1.0, 1.0, 101.0],
        'Coupon_date_dt': [  # make sure all values are datetime objects
            dt.datetime(2022, 12, 31),
            dt.datetime(2023, 12, 31),
            dt.datetime(2024, 12, 31),
            dt.datetime(2025, 12, 31),
            dt.datetime(2026, 12, 31),
        ]
    }

    bond_rate_dict = {
        1.0: 3.5,
        2.0: 3.6,
        3.0: 3.7,
        4.0: 3.8,
        5.0: 4.0
    }

    # Pricing inputs
    s = 12.0  # stock price
    t = dt.datetime(2023, 4, 1)  # current pricing date
    implied_vol = 0.25  # 25% volatility
    r = 0.03  # 3% risk-free rate

    # Run pricing
    price, delta = bs_cb(s, term_example, t, implied_vol, r, bond_rate_dict)

    print("\n=== Convertible Bond Pricing Test ===")
    print(f"Input stock price: {s}")
    print(f"Implied volatility: {implied_vol}")
    print(f"Pure bond value: {calculate_bond(term_example, t, bond_rate_dict):.4f}")
    print(f"Convertible bond price (with option): {price:.4f}")
    print(f"Call option delta: {delta:.4f}")
