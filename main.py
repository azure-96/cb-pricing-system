"""
Main script for convertible bond pricing from preprocessing to pricing.
"""

import datetime as dt
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import multiprocessing

from config_loader import load_config
from preprocessing import run_preprocessing
from pricing_methods.pricing_engine import price_convertible_bond

# ========================== Step 1: Load Config ==========================
config = load_config()

mypath = Path(config["mypath"])
term_path = Path(config["files"]["cb_terms_pickle"])
bond_rate_path = Path(config["files"]["bond_rate_output"])
bond_rate_cn_path = Path(config["files"]["bond_rate_cn_output"])
output_dir = Path(config["cb_pricing_output"])

start_date = int(config["pricing"]["cb_pricing_start_date"])
pricing_models = config["pricing"]["pricing_models"]
vol_list = config["pricing"]["vol_list"]
num_sample = int(config["pricing"]["num_sample"])
plot_result = config["pricing"]["plot_result"]
parallel_mode = config["pricing"]["parallel_mode"]
max_workers = int(config["pricing"]["max_workers"])

colname_mc = [f"MC{v}" for v in vol_list]
colname_bs = [f"BS{v}" for v in vol_list]
colname_ccb = [f"CCB{v}" for v in vol_list]
colname_impliedvol = ["BSImpliedvol", "CCBImpliedvol"]
colname_delta = [f"delta{v}" for v in vol_list] + ["deltaImpliedvol"]
price_col_names = colname_mc + colname_bs + colname_ccb + colname_impliedvol + colname_delta

# ========================== Step 2: Preprocessing ==========================
print("[Preprocessing] Starting preprocessing steps.")
# merged_df = run_preprocessing()  # This runs your preprocessing steps
print("[Preprocessing] Done.\n")

# ========================== Step 3: Load Pricing Inputs ====================
with open(term_path, "rb") as f:
    cb_terms = pickle.load(f)

bond_rate = pd.read_pickle(bond_rate_path)
bond_rate_cn = pd.read_pickle(bond_rate_cn_path)

current_time = dt.datetime.now().strftime("%Y%m%d%H%M%S")


# ========================== Step 4: Pricing Wrapper ==========================
def pricing_wrapper(args):
    return price_convertible_bond(
            args,
            start_date=start_date,
            vol_list=vol_list,
            num_sample=num_sample,
            bond_rate=bond_rate,
            bond_rate_cn=bond_rate_cn,
            price_col_names=price_col_names,
            output_path=output_dir,
            plot_figure=plot_result,
            current_time=current_time,
            selected_models=pricing_models
    )


# ========================== Step 5: Execute Pricing ==========================
if __name__ == "__main__":
    print(f"[Pricing] Pricing {len(cb_terms)} convertible bonds.\n")

    if parallel_mode == "thread":
        with tqdm(total=len(cb_terms)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(pricing_wrapper, args) for args in cb_terms.items()]
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)

    elif parallel_mode == "process":
        with multiprocessing.Pool(processes=max_workers) as pool:
            with tqdm(total=len(cb_terms)) as pbar:
                for _ in pool.imap_unordered(pricing_wrapper, cb_terms.items()):
                    pbar.update(1)

    else:  # Sequential mode
        for args in tqdm(cb_terms.items()):
            pricing_wrapper(args)

    print("\n[Done] Convertible bond pricing complete.")
