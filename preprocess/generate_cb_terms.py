# -*- coding: utf-8 -*-
"""Generate or load convertible bond coupon terms from raw data.

This script builds a dictionary mapping each bond to its coupon schedule,
including annual coupon rates, payment dates, and total term length.
"""

import os
import pickle
import pandas as pd
from tqdm import tqdm
from config_loader import load_config


def load_or_create_cb_terms() -> dict:
    """
    Load cached CB terms dict if exists; otherwise, parse and construct from CSV.

    Returns:
        dict: A dictionary containing coupon terms for each bond, in the format:
            {
                '110003': {
                    'code': '110003.SH',
                    'Coupon_rate': [1.5, 1.8, 2.1, 2.4, 2.8],
                    'Coupon': [1.5, 1.8, 2.1, 2.4, 107.0],
                    'date': ['2009-08-21', '2010-08-21', '2011-08-21', '2012-08-21', '2013-08-21'],
                    'numYears': 5
                },
                ...
            }
    """
    config = load_config()
    bond_data_path = config['bond_data']
    output_file = os.path.join(config['data_path'], 'ConvertibleBondsTerms.pkl')

    # if os.path.exists(output_file):
    #     print("Loaded existing Convertible Bond terms dictionary.")
    #     with open(output_file, 'rb') as f:
    #         return pickle.load(f)

    print("Terms file not found. Generating bond terms from raw CSV...")

    # Read raw coupon schedule data
    filepath = os.path.join(bond_data_path, 'WIND_ConvertibleBondsCoupon.csv')
    df = pd.read_csv(filepath, encoding='utf-8')[
        ['wind_code', 'cash_flows_date', 'cash_flows_per_cny100_par', 'cf_type', 'coupon_rate']
    ]
    df.sort_values(by=['wind_code', 'cash_flows_date'], inplace=True)

    # Build dictionary: bond_code → coupon details
    CBterms = {}
    grouped = df.groupby('wind_code')

    for code, group in tqdm(grouped, desc="Processing CB terms"):
        rates = group['coupon_rate'].tolist()
        cash_flows = group['cash_flows_per_cny100_par'].tolist()
        dates = group['cash_flows_date'].tolist()
        num_years = len(dates)

        CBterms[code.split('.')[0]] = {
            'code': code,
            'Coupon_rate': rates,
            'Coupon': cash_flows,
            'date': dates,
            'numYears': num_years
        }

    # Save to pickle
    with open(output_file, 'wb') as f:
        pickle.dump(CBterms, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Convertible Bond terms dictionary generated and saved.")
    return CBterms


if __name__ == '__main__':
    cb_terms = load_or_create_cb_terms()
    example_key = list(cb_terms.keys())[0]
    print(f"Example: {example_key} → {cb_terms[example_key]}")
