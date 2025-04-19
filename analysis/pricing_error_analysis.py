"""
Convertible Bond Pricing Error Analysis

This module performs grouped analysis of pricing bias between market prices
and theoretical prices calculated using BS and CCB models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config_loader import load_config


def load_result_data(result_dir):
    """Load all pricing result .pkl files into one DataFrame"""
    df_all = pd.DataFrame()
    for file in os.listdir(result_dir):
        if file.endswith('.pkl'):
            df = pd.read_pickle(os.path.join(result_dir, file))
            df_all = pd.concat([df_all, df], axis=0)
    return df_all


def compute_bias_metrics(df, vol_labels):
    """Compute multiple forms of pricing bias metrics"""
    for vol in vol_labels:
        df[f'BiasRate1_BS{vol}'] = df['CLOSE_DP'] / df[f'BS{vol}'] - 1
        df[f'BiasRate1_CCB{vol}'] = df['CLOSE_DP'] / df[f'CCB{vol}'] - 1
        df[f'BiasRate2_BS{vol}'] = df[f'BS{vol}'] / df['CLOSE_DP'] - 1
        df[f'BiasRate2_CCB{vol}'] = df[f'CCB{vol}'] / df['CLOSE_DP'] - 1
        df[f'AbsBiasRate1_BS{vol}'] = df[f'BiasRate1_BS{vol}'].abs()
        df[f'AbsBiasRate1_CCB{vol}'] = df[f'BiasRate1_CCB{vol}'].abs()
        df[f'AbsBiasRate2_BS{vol}'] = df[f'BiasRate2_BS{vol}'].abs()
        df[f'AbsBiasRate2_CCB{vol}'] = df[f'BiasRate2_CCB{vol}'].abs()
    return df


def split_by_level(data, output_path, name, lower = -np.inf, upper = np.inf):
    """Split dataset by pricing level and save to disk"""
    subset = data[(data['level'] >= lower) & (data['level'] < upper)]
    subset.to_pickle(os.path.join(output_path, f'{name}.pkl'))
    return subset


def describe_and_plot(data, factor, output_dir, date_col = 'TRADE_DT', code_col = 'TRADE_CODE', quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]):
    """Generate grouped statistics and time-series plots for a given factor"""
    group_name = f'{factor}_group'
    summary_total = data[factor].describe()

    group_label = []
    for date, daily in data.groupby(date_col):
        daily = daily.reset_index(drop=True)
        labels = pd.qcut(daily[factor].rank(method='first', pct=True), q=quantiles, labels=False)
        daily[group_name] = labels
        group_label.append(daily[[date_col, code_col, group_name]])

    label_df = pd.concat(group_label)
    data = pd.merge(data, label_df, on=[date_col, code_col], how='left')

    summary_group = data.groupby(group_name)[factor].describe().T
    summary = pd.concat([summary_total, summary_group], axis=1)
    summary.columns = ['Total'] + [f'Group {i + 1}' for i in range(5)]

    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, f'Describe_{factor}.csv'))
    plot_grouped_data(data, factor, os.path.join(output_dir, f'{factor}.png'), date_col)


def plot_grouped_data(data, factor, fig_path, date_col = 'TRADE_DT', label_step = 40):
    """Plot overall and grouped median/mean trends"""
    data[date_col] = data[date_col].astype(str)
    group_col = f'{factor}_group'
    grouped = data.groupby(date_col)
    dates = grouped[factor].mean().index

    fig, axes = plt.subplots(3, 1, figsize=(32, 24), dpi=300)

    # Plot overall trend
    axes[0].plot(grouped[factor].median(), label='Median')
    axes[0].plot(grouped[factor].mean(), label='Mean', linestyle='--')
    axes[0].set_title(f'Daily {factor} Trend')
    axes[0].legend()

    # Plot grouped trend
    for i, grp in data.groupby(group_col):
        sub = grp.groupby(date_col)
        axes[1].plot(sub[factor].median(), label=f'Group {int(i)}')
        axes[2].plot(sub[factor].mean(), label=f'Group {int(i)}')

    for ax in axes:
        ax.set_xticks(dates[::label_step])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)
        ax.legend()

    axes[1].set_title(f'Median {factor} by Group')
    axes[2].set_title(f'Mean {factor} by Group')

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close(fig)


def main():
    """Main entry: compute and analyze pricing bias"""
    config = load_config()
    mypath = config['mypath']
    raw_result_dir = config['raw_result_dir']  # Folder with BS/CCB pricing outputs
    error_output_root = config['result_path']  # Root for saving processed analysis results

    df_all = load_result_data(raw_result_dir)
    df_all['level'] = df_all['CLOSE_DP'] / df_all['BondPrice']

    vol_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 250]
    vol_labels = [str(v) for v in vol_list] + ['Impliedvol']
    df_all = compute_bias_metrics(df_all, vol_labels)

    # Save and split by pricing level
    df_all.to_pickle(os.path.join(error_output_root, 'cb_estimator.pkl'))
    df_low = split_by_level(df_all, error_output_root, 'cb_estimator_l', upper=0.8)
    df_mid = split_by_level(df_all, error_output_root, 'cb_estimator_m', lower=0.8, upper=1.2)
    df_high = split_by_level(df_all, error_output_root, 'cb_estimator_h', lower=1.2)

    # Analyze key bias metrics
    test_list = [f'BiasRate1_BS{v}' for v in vol_labels] + [f'BiasRate1_CCB{v}' for v in vol_labels]
    for factor in test_list:
        describe_and_plot(df_all, factor, os.path.join(error_output_root, 'Error'))
        describe_and_plot(df_high, factor, os.path.join(error_output_root, 'Error', 'High'))
        describe_and_plot(df_mid, factor, os.path.join(error_output_root, 'Error', 'Medium'))


if __name__ == '__main__':
    main()
