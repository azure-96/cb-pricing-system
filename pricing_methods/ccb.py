"""
Callable Convertible Bond (CCB) Pricing Module.

This module implements the pricing formula for callable convertible bonds
based on soft call/put constraints.

TODO:
    Add advanced versions for soft-callable and soft-puttable PCCBs
"""

import numpy as np
from scipy.stats import norm
import datetime as dt
from config_loader import load_config

# Load constants from config
_config = load_config()
pricing_cfg = _config["pricing"]
float_lower_bound = pricing_cfg["float_lower_bound"]
float_upper_bound = pricing_cfg["float_upper_bound"]
days_of_year = pricing_cfg["days_of_year"]


def ccb_cb(s: float, term: dict, t: dt.datetime, volatility: float, r: float) -> float:
    """
    Price a Callable Convertible Bond (CCB) using the CCB formula.

    Args:
        s (float): Current stock price
        term (dict): CB term structure, same format as used in bs.py
        t (datetime): Current pricing date
        volatility (float): Stock volatility
        r (float): Risk-free rate

    Returns:
        float: Fair value of the callable convertible bond
    """
    tau = (term['Maturity'] - t).days / days_of_year
    P1 = term['ConvPrice']
    BC = term['Coupon'][-1]
    BF = 100
    P2 = P1 * term['CallOptionTriggerProportion'] / 100
    CN = term['Coupon_rate'][-1]

    idx_i = [i for i, d in enumerate(term['Coupon_date_dt']) if (t < d)]
    tau_i = np.array([(term['Coupon_date_dt'][i] - t).days / days_of_year for i in idx_i])
    coupon_i = np.array([term['Coupon_rate'][i] for i in idx_i])
    Pv = np.sum(coupon_i * np.exp(-r * tau_i))

    mu_bar = r - 0.5 * volatility**2
    mu_hat = r + 0.5 * volatility**2
    mu_tilde = np.sqrt(mu_bar**2 + 2 * r * volatility**2)

    coupon_i = np.array([term['Coupon'][i] for i in idx_i])
    price_bond = np.sum(coupon_i * np.exp(-r * tau_i))

    # Option-related expressions
    a1 = (np.log(P2 / s) + mu_tilde * tau) / (volatility * np.sqrt(tau))
    a2 = (np.log(P2 / s) - mu_tilde * tau) / (volatility * np.sqrt(tau))
    d1 = (np.log(s / ((1 + CN / BF) * P1 * np.exp(-r * tau))) + mu_hat * tau) / (volatility * np.sqrt(tau))
    d2 = (np.log(s / P2) + mu_hat * tau) / (volatility * np.sqrt(tau))
    d3 = (np.log(P2**2 / (s * (1 + CN / BF) * P1 * np.exp(-r * tau))) + mu_hat * tau) / (volatility * np.sqrt(tau))
    d4 = (np.log(P2 / s) + mu_hat * tau) / (volatility * np.sqrt(tau))
    a3 = (np.log(P2 / s) + mu_bar * tau) / (volatility * np.sqrt(tau))
    a4 = (np.log(P2 / s) - mu_bar * tau) / (volatility * np.sqrt(tau))

    all_cdf = norm.cdf([-a1, -a2, d1, d1 - volatility * np.sqrt(tau), d2, d2 - volatility * np.sqrt(tau),
                        -d3, -d3 + volatility * np.sqrt(tau), -d4, -d4 + volatility * np.sqrt(tau),
                        -a3, -a4])
    all_cdf = np.where(all_cdf > float_lower_bound, all_cdf, float_lower_bound)

    power_term_bar_tilde = safe_pow(P2 / s, (mu_bar + mu_tilde) / volatility**2)
    power_term_hat = safe_pow(P2 / s, 2 * mu_hat / volatility**2)
    power_term_bar = safe_pow(P2 / s, 2 * mu_bar / volatility**2)

    ABCi = power_term_bar_tilde * all_cdf[0] + safe_pow(P2 / s, (mu_bar - mu_tilde) / volatility**2) * all_cdf[1]
    ABCi_P2P1 = (P2 - P1) * ABCi

    UOC1 = s * all_cdf[2] - (1 + CN / BF) * P1 * np.exp(-r * tau) * all_cdf[3]
    UOC2 = s * all_cdf[4] - (1 + CN / BF) * P1 * np.exp(-r * tau) * all_cdf[5]
    UOC3 = s * power_term_hat * all_cdf[6] - (1 + CN / BF) * P1 * np.exp(-r * tau) * power_term_bar * all_cdf[7]
    UOC4 = s * power_term_hat * all_cdf[8] - (1 + CN / BF) * P1 * np.exp(-r * tau) * power_term_bar * all_cdf[9]
    UOC = UOC1 - UOC2 + UOC3 - UOC4

    ABCd = power_term_bar * all_cdf[10] + all_cdf[11]
    ABCd_BF = BF * np.exp(-r * tau) * ABCd
    ABCi_BF = BF * ABCi
    ABCd_Fv_T = Pv * ABCd

    if len(idx_i) <= 1:
        ABCi_Fv_tau = 0
    else:
        idx_i_temp = idx_i[:-1]
        tau_i = np.array([(term['Coupon_date_dt'][i] - t).days / days_of_year for i in idx_i_temp])
        r_i = np.array([term['Coupon_rate'][i] for i in idx_i_temp]) / 100.0
        a5 = (np.log(P2 / s) + mu_bar * tau_i) / (volatility * np.sqrt(tau_i))
        a6 = (np.log(P2 / s) - mu_bar * tau_i) / (volatility * np.sqrt(tau_i))
        cdf_t = norm.cdf([-a5, -a6])
        cdf_t = np.where(cdf_t > float_lower_bound, cdf_t, float_lower_bound)
        termt = power_term_bar * all_cdf[10] + all_cdf[11] - (power_term_bar * cdf_t[0, :] + cdf_t[1, :])
        ABCi_Fv_tau = np.sum(BF * r_i * np.exp(-r * tau_i) * termt)

    price_cb = (BF / P1) * ABCi_P2P1 + (BF / P1) * UOC + ABCi_BF - ABCd_BF + ABCi_Fv_tau - ABCd_Fv_T + price_bond
    return price_cb


def safe_pow(base, exponent):
    """Safe exponential power with bounds."""
    try:
        val = base**exponent
    except FloatingPointError:
        return float_upper_bound if exponent > 0 else float_lower_bound
    return min(max(val, float_lower_bound), float_upper_bound)


# -------------------------
# Additional theoretical functions (retained for future work)
# -------------------------

# def EA1(S0, tp, r_f, volatility, h):
#     """Expectation of early conversion under constant volatility."""
#     d = (np.log(S0 / h) + (r_f + 0.5 * volatility**2) * tp) / (volatility * np.sqrt(tp))
#     return S0 * np.exp(r_f * tp) * norm.cdf(d)

# def EA2(S0, tp, tm, volatility, u, tilde_u, K2, h):
#     tp = tm - tp
#     return (h / S0)**((u - tilde_u) / volatility**2) * (
#         fun_G(tm, volatility, tilde_u, K2) - fun_G(tp, volatility, tilde_u, K2) + fun_H(tp, tm, volatility, tilde_u, K2)
#     )

# def EA3(S0, tp, tm, r_f, volatility, hat_u, K1, K2):
# 	d11 = (K2 - hat_u * tp) / (volatility * np.sqrt(tp))
# 	d12 = (K2 - hat_u * tm) / (volatility * np.sqrt(tm))
# 	rho = np.sqrt(tp / tm)
# 	d21 = (K2 + hat_u * tp) / (volatility * np.sqrt(tp))
# 	d22 = (-K2 - hat_u * tm) / (volatility * np.sqrt(tm))
# 	d31 = (K2 - hat_u * tp) / (volatility * np.sqrt(tp))
# 	d32 = (K2 - hat_u * tm) / (volatility * np.sqrt(tm))
# 	d32 = (K1 - hat_u * tm) / (volatility * np.sqrt(tm))
# 	d41 = (K2 + hat_u * tp) / (volatility * np.sqrt(tp))
# 	d42 = (K1 - 2 * K2 - hat_u * tm) / (volatility * np.sqrt(tm))
# 	term1 = multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([d11, d12])
# 	term2 = -np.exp(2 * hat_u * K2 / volatility**2) * multivariate_normal([0, 0], [[1, -rho], [-rho, 1]]).cdf([d21, d22])
# 	term3 = -multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([d31, d32])
# 	term4 = np.exp(2 * hat_u * K2 / volatility**2) * multivariate_normal([0, 0], [[1, -rho], [-rho, 1]]).cdf([d41, d42])
# 	return S0 * np.exp(r_f * tm) * (term1 + term2 + term3 + term4)

# def EA4(u, tp, tm, volatility, K1, K2):
# 	d11 = (K2 - u * tp) / (volatility * np.sqrt(tp))
# 	d12 = (K1 - u * tm) / (volatility * np.sqrt(tm))
# 	rho = np.sqrt(tp / tm)
# 	d21 = (K2 + u * tp) / (volatility * np.sqrt(tp))
# 	d22 = (K1 - 2 * K2 - u * tm) / (volatility * np.sqrt(tm))
# 	term1 = multivariate_normal([0, 0], [[1, rho], [rho, 1]]).cdf([d11, d12])
# 	term2 = -np.exp(2 * u * K2 / volatility**2) * multivariate_normal([0, 0], [[1, -rho], [-rho, 1]]).cdf([d21, d22])
# 	return term1 + term2

# def fun_G(t, volatility, tilde_u, K2):
#     d1 = (-K2 + tilde_u * t) / (volatility * np.sqrt(t))
#     d2 = (-K2 - tilde_u * t) / (volatility * np.sqrt(t))
#     return norm.cdf(d1) + np.exp(2 * tilde_u * K2 / volatility**2) * norm.cdf(d2)

# def fun_H(t1, t2, volatility, tilde_u, K2):
#     d11 = (-K2 + tilde_u * t1) / (volatility * np.sqrt(t1))
#     d12 = (K2 - tilde_u * t2) / (volatility * np.sqrt(t2))
#     rho = -np.sqrt(t1 / t2)
#     d21 = (-K2 - tilde_u * t1) / (volatility * np.sqrt(t1))
#     d22 = (K2 + tilde_u * t2) / (volatility * np.sqrt(t2))
#     term1 = norm.cdf(d11) * norm.cdf(d12)
#     term2 = np.exp(2 * tilde_u * K2 / volatility**2) * norm.cdf(d21) * norm.cdf(d22)
#     return term1 + term2

# def p_in(tau, tp, volatility, u, K2):
#     d11 = (K2 - u * tp) / (volatility * np.sqrt(tp))
#     d12 = (K2 - u * tau) / (volatility * np.sqrt(tau))
#     rho = np.sqrt(tp / tau)
#     d21 = (K2 + u * tp) / (volatility * np.sqrt(tp))
#     d22 = (-K2 - u * tau) / (volatility * np.sqrt(tau))
#     term1 = norm.cdf(d11) * norm.cdf(d12)
#     term2 = -np.exp(2 * u * K2 / volatility**2) * norm.cdf(d21) * norm.cdf(d22)
#     return term1 + term2

if __name__ == '__main__':
    print("=== Callable Convertible Bond Pricing Test ===")

    # Example test term
    t = dt.datetime(2025, 1, 1)
    term_example = {
        'ConvPrice': 10.0,
        'Maturity': dt.datetime(2029, 1, 1),
        'CallOptionTriggerProportion': 130.0,
        'Coupon': [1.0, 1.0, 1.0, 1.0, 101.0],  # Coupon + final principal
        'Coupon_rate': [1.0, 1.0, 1.0, 1.0, 1.0],
        'Coupon_date_dt': [
            dt.datetime(2026, 1, 1),
            dt.datetime(2027, 1, 1),
            dt.datetime(2028, 1, 1),
            dt.datetime(2029, 1, 1),
            dt.datetime(2030, 1, 1),  # Just for demo (not paid in reality)
        ]
    }

    S0 = 12.0  # current stock price
    vol = 0.25  # volatility
    rf = 0.03  # risk-free rate

    price = ccb_cb(s=S0, term=term_example, t=t, volatility=vol, r=rf)
    print(f"Stock Price: {S0}")
    print(f"Volatility: {vol}")
    print(f"Risk-Free Rate: {rf}")
    print(f"Callable CB Price: {price:.4f}")
