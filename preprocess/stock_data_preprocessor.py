"""
Preprocess Stock Data:
1. Split raw stock price data into year-wise files.
2. Create duplicated synthetic data for stress testing.

Reads configuration from config.yaml.
"""

import os
import pandas as pd
from pathlib import Path
from config_loader import load_config


def split_stock_data_by_year(input_file: Path, output_folder: Path) -> None:
    """
    Split raw stock data by year and save each year as CSV and Pickle files.
    """
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"[Error] Failed to read: {input_file}\n{e}")
        return

    try:
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    except Exception as e:
        print("[Error] TRADE_DT datetime conversion failed.\n", e)
        return

    df['Year'] = df['TRADE_DT'].dt.year
    output_folder.mkdir(parents=True, exist_ok=True)

    for year, group in df.groupby('Year'):
        year_df = group.drop(columns=['Year'])
        csv_path = output_folder / f'stocks_{year}.csv'
        pkl_path = output_folder / f'stocks_{year}.pkl'

        n_stocks = year_df['S_INFO_WINDCODE'].nunique()
        n_dates = year_df['TRADE_DT'].nunique()
        print(f"[Info] Year {year}: {n_stocks} stocks, {n_dates} dates.")

        if csv_path.exists() and pkl_path.exists():
            print(f"[Skip] Year {year} already saved.\n")
            continue

        try:
            year_df.to_csv(csv_path, index=False)
            year_df.to_pickle(pkl_path)
            print(f"[Saved] CSV: {csv_path}\n[Saved] PKL: {pkl_path}\n")
        except Exception as e:
            print(f"[Error] Failed to save year {year}: {e}")


def duplicate_stock_data(input_file: Path, output_file: Path, times: int = 8) -> None:
    """
    Duplicate all rows in the dataset 'times' times and save.
    Useful for testing on large datasets.
    """
    try:
        df = pd.read_csv(input_file)
        df_duplicated = pd.concat([df] * times, ignore_index=True)
        df_duplicated.to_csv(output_file, index=False)
        print(f"[Success] Duplicated file saved: {output_file}")
    except Exception as e:
        print(f"[Error] Failed to duplicate data: {e}")


def main():
    config = load_config()
    files = config['files']

    input_csv = Path(files['stock_price_csv'])
    clean_dir = Path(files['stock_clean_dir'])
    duplicated_output = Path(files['duplicated_stock_file'])
    duplicate_times = int(files['duplicate_times'])

    # Step 1: Split raw data
    split_stock_data_by_year(input_csv, clean_dir)

    # Step 2: Generate duplicate for a specific year
    original_2019 = clean_dir / 'stocks_2019.csv'
    duplicate_stock_data(original_2019, duplicated_output, times=duplicate_times)


if __name__ == '__main__':
    main()
