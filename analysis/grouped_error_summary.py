"""
Grouped Error Summary Analysis

This script evaluates pricing bias from convertible bond models (e.g., CCB, BS),
and visualizes factor grouping summaries over time.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config_loader import load_config

pd.set_option('display.max_columns', None)


def describe_data(df, factor, output_path, date_col = "TRADE_DT", code_col = "TRADE_CODE", quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]):
    """
    Group factor into quantiles per date and compute descriptive statistics.
    """
    group_col = f"{factor}_group"
    overall_stats = df[factor].describe()

    grouped = df.groupby(date_col)
    labels = []
    for date, group in grouped:
        group[group_col] = pd.qcut(group[factor].rank(method='first', pct=True), q=quantiles, labels=False)
        labels.append(group[[date_col, code_col, group_col]])

    label_df = pd.concat(labels, ignore_index=True)
    df = pd.merge(df, label_df, on=[date_col, code_col], how='left')

    summary_by_group = df.groupby(group_col)[factor].describe()
    combined = pd.concat([overall_stats, summary_by_group.T], axis=1)
    combined.columns = ["Total"] + [f"Group {i + 1}" for i in range(len(summary_by_group))]

    os.makedirs(output_path, exist_ok=True)
    combined.to_csv(os.path.join(output_path, f"Describe_{factor}.csv"))
    print(combined)

    plot_grouped_data(df, factor, os.path.join(output_path, f"{factor}.png"), date_col)


def plot_grouped_data(df, factor, fig_path, date_col = "TRADE_DT", label_step = 40):
    """Plot grouped factor median and mean trends over time."""
    df[date_col] = df[date_col].astype(str)
    group_col = f"{factor}_group"
    grouped_date = df.groupby(date_col)
    date_index = grouped_date[factor].mean().index

    fig, axes = plt.subplots(3, 1, figsize=(32, 24), dpi=300)

    # Overall
    axes[0].plot(grouped_date[factor].median(), label='Median')
    axes[0].plot(grouped_date[factor].mean(), label='Mean', linestyle='--')
    axes[0].set_title(f"Overall Daily Trend - {factor}")
    axes[0].legend()

    # Grouped trends
    line_styles = ['-', '--', '-.', ':', '-']
    for i, group in df.groupby(group_col):
        if pd.isna(i):
            continue
        median_trend = group.groupby(date_col)[factor].median()
        mean_trend = group.groupby(date_col)[factor].mean()
        axes[1].plot(median_trend, label=f"Group {int(i)}", linestyle=line_styles[int(i)])
        axes[2].plot(mean_trend, label=f"Group {int(i)}", linestyle=line_styles[int(i)])

    for ax in axes:
        ax.set_xticks(date_index[::label_step])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend()

    axes[1].set_title(f"Median Daily Trend by Group - {factor}")
    axes[2].set_title(f"Mean Daily Trend by Group - {factor}")

    plt.tight_layout()
    fig.savefig(fig_path)


if __name__ == '__main__':
    config = load_config()
    result_path = config["result_path"]
    pricing_data_path = config["pricing_data"]
    error_output_path = config["error_output"]

    concat_price_data = True
    if concat_price_data:
        df_all = []
        for filename in os.listdir(result_path):
            if filename.endswith('.pkl') and len(filename) == 10:
                df_temp = pd.read_pickle(os.path.join(result_path, filename))
                df_all.append(df_temp)
        df_pricing = pd.concat(df_all, ignore_index=True)
        df_pricing.to_pickle(pricing_data_path)
    else:
        df_pricing = pd.read_pickle(pricing_data_path)

    model_cols = ['CCB250', 'BS250', 'CCB_impliedvol', 'BS_impliedvol']
    for col in model_cols:
        df_pricing[f'BiasRate_{col}'] = df_pricing[col] / df_pricing['CLOSE_DP'] - 1
        df_pricing[f'Bias_{col}'] = df_pricing[col] - df_pricing['CLOSE_DP']

    df_pricing['level'] = df_pricing['CLOSE_DP'] / df_pricing['BondPrice']
    df_high = df_pricing[df_pricing['level'] >= 1.2]
    df_mid = df_pricing[(df_pricing['level'] >= 0.8) & (df_pricing['level'] < 1.2)]
    df_low = df_pricing[df_pricing['level'] < 0.8]

    df_pricing['daily_mean'] = df_pricing.groupby('TRADE_DT')['CLOSE_DP'].transform('mean')

    output_base = error_output_path
    test_cols = [f'BiasRate_{col}' for col in model_cols]
    for factor in test_cols:
        describe_data(df_pricing, factor, os.path.join(output_base))
        describe_data(df_high, factor, os.path.join(output_base, 'High'))
        describe_data(df_mid, factor, os.path.join(output_base, 'Medium'))
        describe_data(df_low, factor, os.path.join(output_base, 'Low'))
