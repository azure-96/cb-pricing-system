# Convertible Bond Pricing System

This repository provides a modular and extensible framework for pricing convertible bonds (CBs) using multiple models including Black-Scholes (BS), Callable Convertible Bond (CCB), and Monte Carlo (MC) simulation. It is designed to allow easy customization and integration with additional models or datasets.


- Black-Scholes (BS)
- Monte Carlo Simulation (MC)
- Callable Convertible Bond (CCB)

The project supports full data preprocessing, pricing computation, parallel execution, and visualization of pricing results.


## Pricing Pipeline

```
Raw Data (CB, Stock, Bond Yield)
        ↓
Preprocessing (terms, volatilities, yields)
        ↓
Pricing Engine (BS, CCB, MC)
        ↓
Output Results (PKL, CSV, PNG)
```


### `cb_terms` Example (Full)

Each convertible bond is represented by a `cb_code` and corresponding term structure as follows:

```python
{
    '113011': {
        'ConvPrice': 6.84,
        'Maturity': datetime.datetime(2024, 6, 27, 0, 0),
        'ConvertStart': 5.5,
        'Coupon': [0.4, 0.6, 1.0, 1.5, 1.8, 108.0],
        'Coupon_rate': [0.4, 0.6, 1.0, 1.5, 1.8, 2.0],
        'Coupon_date': [
            '2019-06-27',
            '2020-06-27',
            '2021-06-27',
            '2022-06-27',
            '2023-06-27',
            '2024-06-27'
        ],
        'Coupon_date_dt': [
            datetime.datetime(2019, 6, 27),
            datetime.datetime(2020, 6, 27),
            datetime.datetime(2021, 6, 27),
            datetime.datetime(2022, 6, 27),
            datetime.datetime(2023, 6, 27),
            datetime.datetime(2024, 6, 27)
        ],
        'Recall': [5.5, 20, 30, 8.892],
        'CallOptionTriggerProportion': 130.0,
        'Resell': [2.0, 30, 30, 4.788, 103],
        'RedeemTriggerProportion': 70.0
    }
}
```

### Field Explanation

| Field | Description |
|-------|-------------|
| `ConvPrice` | Conversion price of the bond |
| `Maturity` | Maturity date (`datetime`) |
| `ConvertStart` | Years before maturity when conversion starts |
| `Coupon` | Annual coupons (last value includes principal) |
| `Coupon_rate` | Coupon rates (percentage format) |
| `Coupon_date` | Payment dates (as strings) |
| `Coupon_date_dt` | Payment dates (as `datetime`) |
| `Recall` | Callable clause: `[years_before_maturity, trigger_days, observation_window, trigger_price]` |
| `CallOptionTriggerProportion` | Call trigger % of conversion price |
| `Resell` | Put clause: `[years_before_maturity, trigger_days, observation_window, trigger_price, resale_price]` |
| `RedeemTriggerProportion` | Put trigger % of conversion price |


## Configuration (`config.yaml`)

YAML configuration covers:

- Global paths (input/output)
- Pricing model selection (`bs`, `ccb`, `mc`)
- Historical volatility windows
- Monte Carlo settings
- Parallel execution mode
- Debugging test-case support

## Run Instructions

```bash
python main.py
```

This runs preprocessing and then starts the pricing pipeline using the model settings in `config.yaml`.


## Project Structure

```
├── analysis/                # Modules for analyzing pricing outputs and errors
├── preprocess/              # All preprocessing scripts
├── pricing_methods/         # Pricing model implementations (BS, CCB, MC)
├── data/                    # Raw and processed data
├── results/                 # Output results (PKL, figures)
├── LICENSE                  
├── README.md                # Project overview and instructions
├── config.yaml              # Centralized configuration file
├── config_loader.py         # Utility for loading and resolving YAML configs
├── preprocessing.py         # Main script to run preprocessing pipeline
└── main.py                  # Main entry point to run CB pricing
```


## Dependencies

- Python 3.8+
- NumPy / Pandas / Matplotlib
- SciPy / tqdm / PyYAML


## License
MIT