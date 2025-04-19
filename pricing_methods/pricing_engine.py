import os
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from pricing_methods.mc import mc_cb
from pricing_methods.bs import bs_cb
from pricing_methods.ccb import ccb_cb
import matplotlib.pyplot as plt


def price_convertible_bond(
        args,
        start_date: int,
        vol_list: list,
        num_sample: int,
        bond_rate: pd.DataFrame,
        bond_rate_cn: pd.DataFrame,
        price_col_names: list,
        output_path: Path,
        plot_figure: bool = False,
        current_time: str = '',
        selected_models: list = None
):
    cb_code, term_data = args
    bond_file = f"data/ConvertibleBondsData_byCode/{cb_code}.csv"
    if not os.path.exists(bond_file):
        print(f"[Skip] No data found for CB {cb_code}")
        return

    bond_df = pd.read_csv(bond_file)
    bond_df = bond_df[bond_df['TRADE_DT'] >= start_date]
    if bond_df.shape[0] < 10:
        return

    # Parse key dates from bond data
    try:
        clause_end = pd.to_datetime(bond_df['CLAUSE_CONVERSION_2_SWAPSHAREENDDATE'].str[:10].iloc[0])
        convert_start = pd.to_datetime(bond_df['CLAUSE_CONVERSION_2_SWAPSHARESTARTDATE_x'].str[:10].iloc[0])
        redeem_start = pd.to_datetime(bond_df['CLAUSE_CALLOPTION_CONDITIONALREDEEMSTARTDATE'].str[:10].iloc[0])
        putback_start = pd.to_datetime(bond_df['CLAUSE_PUTOPTION_CONDITIONALPUTBACKSTARTENDDATE'].str[:10].iloc[0])
    except Exception:
        print(f"[Error] Failed to parse dates for CB {cb_code}")
        return

    call_trigger = bond_df['CLAUSE_CALLOPTION_TRIGGERPROPORTION'].iloc[0]
    redeem_trigger = bond_df['CLAUSE_PUTOPTION_REDEEM_TRIGGERPROPORTION'].iloc[0]

    # Construct convertible bond term dictionary
    term = {
        "ConvPrice": 100,
        "Maturity": clause_end + pd.Timedelta(days=1),
        "ConvertStart": round((clause_end - convert_start).days / 365, 1),
        "Coupon": term_data["Coupon"],
        "Coupon_rate": term_data["Coupon_rate"],
        "Coupon_date": term_data["date"],
        "Coupon_date_dt": [dt.datetime.strptime(d, "%Y-%m-%d") for d in term_data["date"]],
        "Recall": [round((clause_end - redeem_start).days / 365, 1), 20, 30, call_trigger],
        "CallOptionTriggerProportion": call_trigger,
        "Resell": [round((clause_end - putback_start).days / 365, 1), 30, 30, redeem_trigger, 103],
        "RedeemTriggerProportion": redeem_trigger
    }
    if term['Coupon_date_dt'][-1] < term['Maturity']:
        term['Coupon_date_dt'] = [d + pd.Timedelta(days=1) for d in term['Coupon_date_dt']]

    pricing_df = bond_df[bond_df["TRADE_DT"] < int(term["Maturity"].strftime("%Y%m%d"))]
    if pricing_df.empty:
        return

    # Initialize pricing result arrays
    price_mc = np.zeros((len(pricing_df), len(vol_list)))
    price_bs = np.zeros((len(pricing_df), len(vol_list)))
    price_ccb = np.zeros((len(pricing_df), len(vol_list)))
    price_imp_vol = np.zeros((len(pricing_df), 2))
    delta = np.zeros((len(pricing_df), len(vol_list) + 1))

    # Loop over each date to compute prices
    for i, (_, row) in enumerate(pricing_df.iterrows()):
        t = int(row["TRADE_DT"])
        t_dt = dt.datetime.strptime(str(t), "%Y%m%d")
        s = row["S_DQ_CLOSE"]
        r = row["BOND_RATE"] / 100
        conv_price = row["CONVPRICE"]
        implied_vol = row["ImpliedVolatility"]
        hist_vols = row[[f"hist_vol_{v}" for v in vol_list]].values.tolist()

        # Update dynamic term fields
        term["ConvPrice"] = conv_price
        term["Recall"][-1] = call_trigger * conv_price / 100
        term["Resell"][-2] = redeem_trigger * conv_price / 100
        term["Resell"][-1] = row["CLAUSE_PUTOPTION_RESELLINGPRICE"]

        tau = (term["Maturity"] - t_dt).days / 365.0
        y = 0.5 if tau <= 0.5 else np.ceil(tau)
        rf = bond_rate_cn.loc[(bond_rate_cn["TRADE_DT"] == t) & (bond_rate_cn["Period"] == y), "BOND_RATE"]
        risk_free_rate = rf.iloc[0] / 100 if not rf.empty else 0.03

        credit_rating = row["CREDITRATING"]
        br_slice = bond_rate[(bond_rate["CREDITRATING"] == credit_rating) & (bond_rate["TRADE_DT"] == t)]
        br_dict = br_slice.set_index("Period")["BOND_RATE"].to_dict()
        br_dict[0.0] = 0.0
        if br_dict == {0.0: 0.0}:
            br_dict = {
                0.0: 0.0,
                1.0: 0.03,
                2.0: 0.031,
                3.0: 0.032,
                4.0: 0.033,
                5.0: 0.034,
                6.0: 0.035
            }

        # Compute prices under different models
        for j, v in enumerate(hist_vols):
            v = min(max(v, 0.01), 1.0)
            if "bs" in selected_models:
                price_bs[i, j], delta[i, j] = bs_cb(s, term, t_dt, v, risk_free_rate, br_dict)
            if "ccb" in selected_models:
                price_ccb[i, j] = ccb_cb(s, term, t_dt, v, r)
            if "mc" in selected_models:
                price_mc[i, j] = mc_cb(s, term, t, v, risk_free_rate, br_dict, num_sample)

        # Use implied volatility for pricing
        if "bs" in selected_models:
            price_imp_vol[i, 0], delta[i, len(vol_list)] = bs_cb(s, term, t_dt, implied_vol, r, br_dict)
        if "ccb" in selected_models:
            price_imp_vol[i, 1] = ccb_cb(s, term, t_dt, implied_vol, r)

    # Construct final result dataframe
    CB_df = pd.DataFrame(np.hstack((price_mc, price_bs, price_ccb, price_imp_vol, delta)), columns=price_col_names)

    base_cols = ["TRADE_DT", "TRADE_CODE", "PRE_CLOSE_DP", "CLOSE_DP", "BondPrice", "ImpliedVolatility"]
    vol_cols = [f"hist_vol_{v}" for v in vol_list] + [f"pre_hist_vol_{v}" for v in vol_list]
    df_merge = pricing_df[base_cols + vol_cols].reset_index(drop=True)
    df_merge = pd.concat([df_merge, CB_df], axis=1)

    # Save pricing result
    save_path = output_path / f"pricing_{cb_code}.pkl"
    df_merge.to_pickle(save_path)
    print(f"[Saved] Pricing results for {cb_code} to {save_path}")

    return None
