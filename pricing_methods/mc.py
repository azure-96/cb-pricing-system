# -*- coding: utf-8 -*-
"""
Monte Carlo Pricing Model for Convertible Bonds

This module implements MC-based pricing for convertible bonds with callable and resell terms.
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
from scipy import sparse
from config_loader import load_config

# load config
config = load_config()
mypath = config['mypath']
calendar_path = config['pricing']['calendar_csv']
trading_of_year = config['pricing']['trading_days_per_year']
mc_simulations = config['pricing']['mc_simulations']

# load trading calendar
trading_date = pd.read_csv(calendar_path)
trading_date_sse = trading_date[trading_date['S_INFO_EXCHMARKET'] == 'SSE']
trading_date_sse = trading_date_sse.sort_values(by='TRADE_DAYS').reset_index(drop=True)


def mc_cb(stock: float, term: dict, now: int, vol: float, r: float,
          bond_rate_dict: dict, num_to_mock_float = mc_simulations) -> float:
    """
    Price convertible bonds using Monte Carlo simulation.

    Steps:
    1. Simulate stock price paths using the Monte Carlo method.
    2. For each simulated path, evaluate the convertible bond value:
        2.1 If any special clauses (e.g., redemption, call/put options) are triggered,
            compute their respective effects on the bond value.
        2.2 Otherwise, calculate the standard convertible bond payoff.
    3. Return the average payoff across all simulated paths as the final bond price.

    Args:
        stock (float): Current stock price.
        term (dict): Terms and conditions of the convertible bond.
        now (int): Pricing date in YYYYMMDD format (e.g., 20210728).
        vol (float): Volatility of the underlying stock.
        r (float): Risk-free interest rate.
        bond_rate_dict (dict): Dictionary of credit rating to bond rate by period.
        num_to_mock (int): Number of Monte Carlo simulations.

    Returns:
        float: Estimated convertible bond price.
    """
    num_to_mock = int(num_to_mock_float)
    t0 = dt.datetime.strptime(str(now), "%Y%m%d")

    tau = ((np.array(term['Coupon_date_dt']) - t0) / dt.timedelta(days=1)).astype(float) / 365.0
    tau_dict = {i: tau[:i] for i in range(len(tau) + 1)}
    period = 0.5 * ((tau <= 0.5) * (tau > 0)) + np.ceil(tau) * (tau > 0.5)
    rate = np.array(list(map(bond_rate_dict.get, period)))
    coupon = np.array(term['Coupon'])

    # prepare timeline
    arr_day_series = get_trading_date_list(now, term['Maturity'])
    num_nodes = len(arr_day_series)
    if num_nodes + 1 < term['Recall'][2]:
        return 0

    arr_time_series = np.array([(term['Maturity'] - date).days for date in [t0] + arr_day_series]) / 365.0
    arr_tau_series = np.array([(date - t0).days for date in arr_day_series]) / 365.0
    period_class = (np.ones((len(arr_tau_series), len(tau))) * tau <= arr_tau_series[:, None]).sum(axis=1)

    # MC simulation
    arr_mc = _monte_carlo(stock, vol, r, num_nodes, num_to_mock)

    # Callable and Resell Trigger
    recall_set = _trigger_event(arr_mc, term['Recall'], arr_time_series, True)
    resell_set = _trigger_event(arr_mc, term['Resell'], arr_time_series, False)

    # build execution decision
    term_coord = np.ones((num_to_mock, 2)) * (num_nodes + 1)
    term_coord[recall_set[0], 0] = recall_set[1]
    term_coord[resell_set[0], 1] = resell_set[1]

    recall_part = sparse.coo_array(((term_coord[:, 0] < term_coord[:, 1]) * term_coord[:, 0]).astype(int))
    resell_part = sparse.coo_array(((term_coord[:, 0] > term_coord[:, 1]) * term_coord[:, 1]).astype(int))
    non_triggered = sparse.coo_array((recall_part + resell_part).toarray() == 0)

    price_recall = cal_term_trigger_price('recall', recall_part.col, recall_part.data, arr_mc, arr_tau_series,
                                          period_class, bond_rate_dict, tau_dict, term)
    price_resell = cal_term_trigger_price('resell', resell_part.col, resell_part.data, arr_mc, arr_tau_series,
                                          period_class, bond_rate_dict, tau_dict, term)

    # final payoff if no trigger
    sub_mc = arr_mc[non_triggered.col, -1]
    final_coupon = np.ones((len(sub_mc), len(coupon))) * coupon
    final_coupon[:, -1] = np.maximum(sub_mc * 100 / term['ConvPrice'], final_coupon[:, -1])

    rate = np.array([np.nan if r is None else r for r in rate], dtype=float)
    rate_clean = np.nan_to_num(rate, nan=0.0)
    price_non = np.sum(np.exp(-tau * rate_clean / 100) * final_coupon * (rate_clean > 0), axis=1)

    return (price_recall.sum() + price_resell.sum() + price_non.sum()) / num_to_mock


def _monte_carlo(stock: float, vol: float, r: float, num_nodes: int, num_to_mock = 100) -> np.ndarray:
    """Simulate stock price paths using geometric Brownian motion"""
    arr_stock = np.ones((num_to_mock, num_nodes + 1))
    dt = 1 / trading_of_year
    rc = np.log(1 + r)
    eps = np.random.normal(0, 1, (num_to_mock, num_nodes))
    arr_mock = np.cumprod(np.exp((rc - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * eps), axis=1)
    arr_stock[:, 1:] = arr_mock
    return arr_stock * stock


def get_trading_date_list(start_date: int, end_date: dt.datetime) -> list:
    """Get trading date list between two dates"""
    end_int = int(end_date.strftime("%Y%m%d"))
    dates = trading_date_sse[(trading_date_sse['TRADE_DAYS'] > start_date) & (trading_date_sse['TRADE_DAYS'] < end_int)]
    return [dt.datetime.strptime(str(d), "%Y%m%d") for d in dates['TRADE_DAYS'].tolist()]


def _trigger_event(price: np.ndarray, term: list, time_series: np.ndarray, is_recall = True) -> np.ndarray:
    """Check and locate first valid trigger event"""
    threshold = term[-1] if is_recall else term[-2]
    flag = (price > threshold) if is_recall else (price < threshold)
    count = _rolling_sum(flag, term[2])
    trigger_mask = (count >= term[1]) & (time_series < term[0])
    coords = np.vstack(np.where(trigger_mask))
    _, first = np.unique(coords[0], return_index=True)
    return coords[:, first]


def _rolling_sum(matrix: np.ndarray, window: int) -> np.ndarray:
    """Efficient rolling sum across axis 1"""
    ret = np.cumsum(matrix, axis=1)
    ret[:, window:] -= ret[:, :-window]
    ret[:, :window - 1] = 0
    return ret


def cal_term_trigger_price(type_: str, rows, cols, mc, tau, period_class, rate_dict, tau_dict, term) -> pd.Series:
    """Calculate discount value at trigger points"""
    spot_price = mc[rows, cols]
    sub_period = (period_class[cols - 1]).astype(int)

    # For each triggered point, get the coupon periods before that time (tau_dict[sp])
    # and append the trigger time (tau[cols[i] - 1]) as the final discount point
    tau_term = [np.append(tau_dict[sp], tau[cols[i] - 1]) for i, sp in enumerate(sub_period)]
    period_term = pd.DataFrame([0.5 * ((x <= 0.5) * (x > 0)) + np.ceil(x) * (x > 0.5) for x in tau_term])

    # rate_term = period_term.applymap(rate_dict.get)
    rate_term = period_term.apply(lambda col: col.map(rate_dict))

    if type_ == 'recall':
        coupon_term = pd.DataFrame([
            np.append(term['Coupon'][:p], spot_price[i] * 100 / term['ConvPrice'])
            for i, p in enumerate(sub_period)
        ])
    else:
        if np.isnan(term['Resell'][-1]):
            last_tau = [x[-1] - x[-2] if len(x) > 1 else x[0] for x in tau_term]
            coupon_term = pd.DataFrame([
                np.append(term['Coupon_rate'][:p], 100 + last_tau[i] * term['Coupon_rate'][p])
                for i, p in enumerate(sub_period)
            ])
        else:
            coupon_term = pd.DataFrame([
                np.append(term['Coupon'][:p], term['Resell'][-1])
                for i, p in enumerate(sub_period)
            ])
    tau_term = pd.DataFrame(tau_term)
    return np.sum(np.exp(-tau_term * rate_term / 100) * coupon_term * (rate_term > 0), axis=1)


if __name__ == '__main__':
    # Minimal example for unit test
    dummy_term = {
        'ConvPrice': 10,
        'Maturity': dt.datetime(2027, 12, 31),
        'Coupon': [1.0, 1.0, 1.0, 1.0, 105.0],
        'Coupon_rate': [1.0] * 5,
        'Coupon_date_dt': [dt.datetime(2024 + i, 12, 31) for i in range(5)],
        'Recall': [5.0, 20, 30, 12.0],
        'Resell': [5.0, 20, 30, 7.0, 103],
    }
    dummy_bond_rate = {1.0: 3.5, 2.0: 3.7, 3.0: 3.8, 4.0: 3.9, 5.0: 4.0}
    price = mc_cb(stock=9.5, term=dummy_term, now=20240418, vol=0.25, r=0.03, bond_rate_dict=dummy_bond_rate)
    print(f"Simulated MC Price: {price:.2f}")
