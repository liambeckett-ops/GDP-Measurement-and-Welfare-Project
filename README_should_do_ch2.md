This folder contains helper scaffolding for the "Should Do Ch2 – GDP, Measurement, and Welfare" Quarto assignment.

Files added:

- `requirements.txt` — Python packages you should install (use `pip install -r requirements.txt`).
- `should_do_ch2_helpers.py` — A small Python module with functions to fetch FRED and World Bank WDI series,
  compute expenditure shares, format the most recent quarters table, and produce plots and simple statistics.

Quick start

1. Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

2. Set your FRED API key as an environment variable (get a free key at https://fred.stlouisfed.org/):

PowerShell

```powershell
$Env:FRED_API_KEY = "YOUR_KEY_HERE"
```

Command Prompt

```cmd
set FRED_API_KEY=YOUR_KEY_HERE
```

3. Example usage inside your Quarto notebook (`should_do_ch2.qmd`):

```python
from should_do_ch2_helpers import fetch_fred_dataframe, compute_expenditure_shares, latest_quarters_table, plot_shares_time_series, EXAMPLE_US_SERIES

# Replace or verify the example FRED codes in EXAMPLE_US_SERIES if necessary.
series_map = EXAMPLE_US_SERIES
# fetch data from 1960 to present
df = fetch_fred_dataframe(series_map, start='1960-01-01')
shares = compute_expenditure_shares(df, gdp_col='GDP')
# show most recent five quarterly shares
latest_table = latest_quarters_table(shares, n=5)
print(latest_table)
# plot long-run shares
plot_shares_time_series(shares, start_year=1960)
```

4. For World Bank indicators (WDI) via `wbdata`:

```python
import wbdata
from should_do_ch2_helpers import fetch_wdi_indicators, scatter_vs_income

# Example indicator mapping: code -> label
indicators = {
    'NY.GDP.PCAP.PP.KD': 'GDP_PPP_per_capita',
    'SP.POP.TOTL': 'Population'
}
# fetch data for 2020
wdi = fetch_wdi_indicators(indicators, countries=None, start_year=2020, end_year=2020)
# prepare a DataFrame indexed by country and date; then plot scatter
stats = scatter_vs_income(wdi.reset_index(), indicator_col='Population', income_col='GDP_PPP_per_capita', year=2020)
print(stats)
```

Notes and verification

- The helper module provides sensible defaults but does not bake in assumptions about FRED series codes. Verify series IDs (especially government and net export codes) on FRED before using.
- If you prefer `pandas_datareader` or another client for World Bank, adapt the functions in `should_do_ch2_helpers.py` accordingly.

If you want, I can now:
- Update `should_do_ch2.qmd` to call these helpers and fill in template code chunks, or
- Run a short test fetching one FRED series to validate the setup (I will need your FRED API key set in the environment of the terminal session).