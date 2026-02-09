"""Helper utilities for the "Should Do Ch2" Quarto assignment.

Usage:
- Set environment variable `FRED_API_KEY` with your FRED API key.
- Install dependencies from `requirements.txt`.
- Import functions into your Quarto notebook and call them to fetch data, compute shares,
  and produce plots/tables.

This module intentionally keeps functions small and well-documented so you can
copy-paste into the Quarto document or call them from a separate analysis script.
"""

from typing import Dict, Optional, Sequence, Tuple
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# External data clients
try:
    from fredapi import Fred
except Exception:
    Fred = None

try:
    import wbdata
except Exception:
    wbdata = None

import statsmodels.api as sm
from scipy import stats


def _get_fred_client(api_key: Optional[str] = None) -> "Fred":
    """Return a fredapi.Fred client using the provided key or environment variable.

    Raises a ValueError if fredapi is not installed or no API key is found.
    """
    if Fred is None:
        raise ImportError("fredapi is not installed. Install with `pip install fredapi`.")
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise ValueError("FRED API key not provided. Set FRED_API_KEY environment variable or pass api_key.")
    return Fred(api_key=key)


def fetch_fred_series(series_id: str, start: Optional[str] = None, end: Optional[str] = None,
                      api_key: Optional[str] = None) -> pd.Series:
    """Fetch a single FRED series as a pandas Series.

    Parameters
    - series_id: FRED series ID (e.g., 'GDP')
    - start, end: date strings parseable by pandas (e.g., '1960-01-01')
    - api_key: optional FRED API key
    """
    client = _get_fred_client(api_key)
    s = client.get_series(series_id, start_date=start, end_date=end)
    s.name = series_id
    return s


def fetch_fred_dataframe(series_map: Dict[str, str], start: Optional[str] = None,
                         end: Optional[str] = None, api_key: Optional[str] = None) -> pd.DataFrame:
    """Fetch multiple FRED series and return a single aligned DataFrame.

    - series_map: mapping of "label" -> "FRED_SERIES_ID" (e.g., {'GDP': 'GDP', 'Consumption':'PCE'})
    """
    client = _get_fred_client(api_key)
    data = {}
    for label, sid in series_map.items():
        data[label] = client.get_series(sid, start_date=start, end_date=end)
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_expenditure_shares(df: pd.DataFrame, gdp_col: str = 'GDP') -> pd.DataFrame:
    """Compute expenditure shares as percentages of nominal GDP.

    Expects `df[gdp_col]` and other component columns to be in the same units.
    Returns a DataFrame of shares (percent).
    """
    if gdp_col not in df.columns:
        raise KeyError(f"gdp_col '{gdp_col}' not found in DataFrame columns")
    shares = df.div(df[gdp_col], axis=0) * 100.0
    # Optional: compute Net exports as X - M if both present
    if ('Exports' in df.columns) and ('Imports' in df.columns):
        shares['NetExports'] = (df['Exports'] - df['Imports']) / df[gdp_col] * 100.0
    return shares


def latest_quarters_table(shares_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return a table (n most recent rows) of shares formatted as percentages with 2 decimals.

    Assumes the DataFrame index is a DatetimeIndex at quarterly frequency.
    """
    # Take last n rows
    last = shares_df.dropna(how='all').iloc[-n:]
    # Format numeric display
    formatted = last.round(2)
    return formatted


def plot_shares_time_series(shares_df: pd.DataFrame, start_year: int = 1960,
                            title: Optional[str] = None, figsize: Tuple[int, int] = (12, 6),
                            savepath: Optional[str] = None) -> None:
    """Plot time series of expenditure shares from start_year to end of series.

    Plots each column in `shares_df` as a percent share over time.
    """
    df = shares_df.copy()
    df = df[df.index.year >= start_year]
    plt.figure(figsize=figsize)
    sns.set_style('whitegrid')
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    plt.xlabel('Year')
    plt.ylabel('Share of Nominal GDP (%)')
    plt.title(title or f'Expenditure Shares since {start_year}')
    plt.legend()
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()


# ----- World Bank / WDI helpers -----

def fetch_wdi_indicators(indicators: Dict[str, str], countries: Optional[Sequence[str]] = None,
                         start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
    """Fetch World Bank WDI indicators using `pandas_datareader` (fallback to `wbdata`).

    - indicators: mapping of code -> label, e.g. {'NY.GDP.PCAP.PP.KD': 'GDP_PPP_per_capita'}
    - countries: list of country ISO2/ISO3 codes or None for all countries
    - start_year/end_year: integer years to limit the date range

    Returns a multi-index DataFrame with index (country, date) and columns = labels.
    """
    try:
        import pandas_datareader.wb as wb
        # wb.download returns DataFrame with country index, indicator columns
        indicator_codes = list(indicators.keys())
        df = wb.download(indicator=indicator_codes, country=countries or 'all', start=start_year, end=end_year)
        # Rename columns to labels
        df = df.rename(columns=indicators)
        # Add date column (since it's yearly, use the year)
        df['date'] = pd.to_datetime(str(start_year)) if start_year else pd.to_datetime('2020')
        # Set multi-index
        df = df.reset_index().set_index(['country', 'date'])
        return df
    except Exception:
        # Fallback to wbdata
        if wbdata is None:
            raise ImportError("wbdata is not installed. Install with `pip install wbdata`.")
        # wbdata.get_dataframe expects indicators as dict code: name
        df = wbdata.get_dataframe(indicators, country=countries, convert_date=False)
        # rename columns to user-friendly labels
        df = df.rename(columns=indicators)
        # Filter by year if specified
        if start_year or end_year:
            df = df[df.index.get_level_values('date') >= str(start_year or 0)]
            if end_year:
                df = df[df.index.get_level_values('date') <= str(end_year)]
        return df


def scatter_vs_income(df: pd.DataFrame, indicator_col: str, income_col: str,
                      year: Optional[int] = None, figsize: Tuple[int, int] = (8, 6),
                      annotate_outliers: bool = True, savepath: Optional[str] = None) -> Dict[str, float]:
    """Create a scatter plot of `indicator_col` vs `income_col` and return stats.

    - df: DataFrame with columns including indicator_col and income_col and a date index or date column
    - year: if provided, filters observations to that year
    Returns a dict with 'pearson_r' and 'p_value' and regression `params`.
    """
    plot_df = df.copy()
    # If multi-index (country, date) or date index, try to filter
    if year is not None:
        if isinstance(plot_df.index, pd.MultiIndex) and 'date' in plot_df.index.names:
            plot_df = plot_df.xs(str(year), level='date', drop_level=False)
        else:
            # try to access a 'date' column or the index
            if 'date' in plot_df.columns:
                plot_df = plot_df[plot_df['date'].dt.year == year]
            else:
                # if index is datetime
                if isinstance(plot_df.index, pd.DatetimeIndex):
                    plot_df = plot_df[plot_df.index.year == year]
    plot_df = plot_df.dropna(subset=[indicator_col, income_col])
    x = plot_df[income_col].astype(float)
    y = plot_df[indicator_col].astype(float)
    # correlation
    pearson_r, p_value = stats.pearsonr(x, y)
    # regression: y ~ x
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    params = model.params.to_dict()

    # plot
    plt.figure(figsize=figsize)
    sns.scatterplot(x=x, y=y)
    sns.regplot(x=x, y=y, scatter=False, color='red', line_kws={'linewidth':1})
    plt.xlabel(income_col)
    plt.ylabel(indicator_col)
    plt.title(f'{indicator_col} vs {income_col}' + (f' ({year})' if year else ''))
    if annotate_outliers and len(plot_df) <= 150:
        # annotate top 3 absolute residuals
        preds = model.predict(X)
        residuals = (y - preds).abs()
        outliers = residuals.nlargest(3).index
        for idx in outliers:
            label = idx[0] if isinstance(idx, tuple) else str(idx)
            plt.annotate(label, (x.loc[idx], y.loc[idx]))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

    return {"pearson_r": pearson_r, "p_value": p_value, "regression_params": params}


# Small utility: example mapping of common U.S. series (verify codes before use)
EXAMPLE_US_SERIES = {
    'GDP': 'GDP',              # Nominal GDP (US)
    'Consumption': 'PCE',      # Personal consumption expenditures
    'Investment': 'GPDI',      # Gross private domestic investment
    'Government': 'GCE',       # Government consumption & investment (verify code)
    'Exports': 'EXPGS',        # Exports of goods and services (verify)
    'Imports': 'IMPGS',        # Imports of goods and services (verify)
}


if __name__ == '__main__':
    print('should_do_ch2_helpers loaded. Import functions into your analysis notebook.')
