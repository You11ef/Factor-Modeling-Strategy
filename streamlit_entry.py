import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import financedatabase as fd
import statsmodels.api as sm
from pandas_datareader import data as web

# -------------------------------------------------
# 1. PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Factor-Based Portfolio (3-Factor)", layout="wide")

# -------------------------------------------------
# 2. HELPER FUNCTIONS & CACHING
# -------------------------------------------------

@st.cache_data
def load_ticker_list():
    """Load & prepare a comprehensive ticker list from financedatabase."""
    df_etfs = fd.ETFs().select().reset_index()[['symbol', 'name']]
    df_equities = fd.Equities().select().reset_index()[['symbol', 'name']]
    df_all = pd.concat([df_etfs, df_equities], ignore_index=True)
    df_all.dropna(subset=['symbol'], inplace=True)
    df_all = df_all[~df_all['symbol'].str.startswith('^')]
    df_all['symbol_name'] = df_all['symbol'] + " - " + df_all['name'].fillna('')
    df_all.drop_duplicates(subset='symbol', inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    return df_all

@st.cache_data
def fetch_prices(tickers, start, end):
    """Download price data once, to avoid re-downloading for every small parameter change."""
    return yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

@st.cache_data
def fetch_ff_3_factors(start, end):
    """
    Fetch Fama-French 3-Factor data via pandas_datareader.
    Columns: Mkt-RF, SMB, HML, RF (all in decimal form, monthly).
    """
    ff_3 = web.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)
    df_factors_raw = ff_3[0].copy()  # the first item is the monthly data (in percent)
    
    # Typically columns: Mkt-RF, SMB, HML, RF
    # Convert from percent to decimal
    df_factors = df_factors_raw / 100.0

    # Convert the index (Period) to a Timestamp at month-end
    df_factors.index = df_factors.index.to_timestamp("M")

    # Keep only Mkt-RF, SMB, HML, RF
    df_factors = df_factors[['Mkt-RF', 'SMB', 'HML', 'RF']]
    
    return df_factors

def portfolio_return(weights, returns_vector):
    """
    Given a vector of weights and an expected return vector, compute the dot product.
    If returns_vector is annual, this result is the expected annual portfolio return.
    """
    return np.dot(weights, returns_vector)

def portfolio_std(weights, cov_matrix):
    """
    Compute the standard deviation using the covariance matrix (annual or monthly).
    """
    return np.sqrt(weights.T @ cov_matrix @ weights)

def sharpe_ratio(weights, expected_returns, cov, rf):
    """
    Sharpe ratio = (Portfolio Return - rf) / Portfolio Vol.
    If 'expected_returns' is annual, 'rf' must be annual, 'cov' must be annual as well.
    """
    mu_p = portfolio_return(weights, expected_returns)
    sigma_p = portfolio_std(weights, cov)
    excess = mu_p - rf
    return excess / sigma_p if sigma_p != 0 else 0

def sortino_ratio(weights, expected_returns, rf, returns_history):
    """
    Sortino ratio = (mu - rf) / downside deviation.
    returns_history should be the monthly (or daily) time series for the selected assets.
    We'll need to replicate what we do for Sharpe, but focusing on negative returns.
    """
    mu_p = portfolio_return(weights, expected_returns)
    portfolio_ts = (returns_history @ weights)  # shape: time x assets => portfolio returns time series
    excess_ts = portfolio_ts - rf  # if rf is a scalar monthly or daily

    # Keep only negative returns
    negative_excess = excess_ts[excess_ts < 0]
    if len(negative_excess) == 0:
        return float('inf')
    downside_std = np.sqrt((negative_excess**2).mean())
    if downside_std == 0:
        return float('inf')
    return (mu_p - rf) / downside_std

def objective_function(weights, expected_returns, cov, rf, obj, returns_history):
    """
    Unified objective for:
    - "Max Sharpe Ratio": minimize negative Sharpe
    - "Min Volatility"  : minimize vol
    - "Max Return"      : minimize negative return
    - "Max Sortino Ratio": minimize negative sortino
    """
    if obj == "Max Sharpe Ratio":
        return -sharpe_ratio(weights, expected_returns, cov, rf)
    elif obj == "Min Volatility":
        return portfolio_std(weights, cov)
    elif obj == "Max Return":
        return -portfolio_return(weights, expected_returns)
    elif obj == "Max Sortino Ratio":
        return -sortino_ratio(weights, expected_returns, rf, returns_history)

@st.cache_data
def generate_random_portfolios(num_portfolios, n_assets, min_weight, max_weight,
                               expected_returns, cov, rf, obj, returns_history):
    """
    Generate random feasible portfolios for plotting a frontier,
    now using the factor-based expected returns for returns, and
    the chosen objective for color dimension.
    """
    results = []
    for _ in range(num_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        if np.any(w < min_weight) or np.any(w > max_weight):
            continue
        mu_p = portfolio_return(w, expected_returns)
        vol_p = portfolio_std(w, cov)
        sr_p = sharpe_ratio(w, expected_returns, cov, rf)
        results.append((mu_p, vol_p, sr_p, w))
    
    df_rand = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe', 'Weights'])
    return df_rand

# -------------------------------------------------
# 3. MAIN APP LAYOUT
# -------------------------------------------------
st.title("Factor-Based Portfolio Optimization (Fama-French 3-Factor)")

st.sidebar.header("Portfolio Settings")

# 3.1 Ticker Selection
ticker_list = load_ticker_list()
st.sidebar.subheader("Choose Tickers")
sel_tickers = st.sidebar.multiselect(
    "Search and Select Tickers",
    options=ticker_list["symbol_name"],
    default=[]
)
sel_symbol_list = ticker_list.loc[ticker_list.symbol_name.isin(sel_tickers), 'symbol'].tolist()

# 3.2 Date Range
st.sidebar.subheader("Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

# We'll stick to Monthly frequency for Fama-French alignment
freq_choice = "Monthly"
ann_factor = 12  # We'll treat everything monthly => multiply by 12 for annual

# 3.3 Risk-Free Rate (Optional Override)
st.sidebar.subheader("Risk-Free Rate")
use_custom_rf = st.sidebar.checkbox("Use Custom RF instead of Fama-French RF?", value=False)
custom_rf_annual = 1.5  # default annual
if use_custom_rf:
    custom_rf_annual = st.sidebar.number_input("Enter Annual Risk-Free Rate (%)", value=1.5)
    st.sidebar.write(f"Custom annual RF = {custom_rf_annual:.2f}%")

# 3.4 Weight Bounds
st.sidebar.subheader("Weight Bounds per Asset")
min_weight = st.sidebar.slider("Minimum Weight", 0.0, 0.5, 0.0, 0.05)
max_weight = st.sidebar.slider("Maximum Weight", 0.0, 1.0, 0.4, 0.05)

# 3.5 Optimization Objective
st.sidebar.subheader("Objective Function")
opt_objective = st.sidebar.selectbox(
    "Optimization Objective",
    ["Max Sharpe Ratio", "Min Volatility", "Max Return", "Max Sortino Ratio"]
)

# -------------------------------------------------
# 4. PRICE DATA & FAMA-FRENCH REGRESSION
# -------------------------------------------------
if sel_symbol_list:
    st.markdown(f"**You have selected {len(sel_symbol_list)} tickers**: {', '.join(sel_symbol_list)}")
    st.subheader("Data Fetch & Factor Regression (Monthly)")

    # 4.1 Fetch daily price data, then convert to monthly
    data_raw = fetch_prices(sel_symbol_list, start=start_date, end=end_date)
    if data_raw.empty:
        st.warning("No data available for the selected tickers and date range.")
    else:
        # Use Close
        if 'Close' in data_raw:
            price_data_raw = data_raw['Close']
        else:
            price_data_raw = data_raw

        # Forward-fill & drop
        price_data_raw = price_data_raw.ffill().dropna(how='all')

        # Identify valid vs. unavailable tickers
        unavailable_tickers = []
        valid_tickers = []
        for t in sel_symbol_list:
            if (t not in price_data_raw.columns) or (price_data_raw[t].isnull().all()):
                unavailable_tickers.append(t)
            else:
                valid_tickers.append(t)

        if unavailable_tickers:
            st.info(
                "The following tickers returned no data in the chosen range "
                f"and will be excluded: {', '.join(unavailable_tickers)}"
            )
        if not valid_tickers:
            st.warning("No valid data available for the selected tickers.")
        else:
            full_price_data = price_data_raw[valid_tickers]
            st.line_chart(full_price_data, height=300)

            # Convert daily to monthly
            monthly_prices = full_price_data.resample('ME').last()
            monthly_returns = monthly_prices.pct_change().dropna()

            # Overlapping date range
            first_valid = full_price_data.apply(lambda col: col.first_valid_index())
            last_valid = full_price_data.apply(lambda col: col.last_valid_index())

            table_data = []
            for t in valid_tickers:
                earliest = first_valid[t]
                latest = last_valid[t]
                table_data.append({
                    "Ticker": t,
                    "Earliest Available": earliest.strftime("%Y-%m-%d") if earliest else "N/A",
                    "Latest Available": latest.strftime("%Y-%m-%d") if latest else "N/A"
                })
            st.subheader("Data Availability Table")
            st.dataframe(pd.DataFrame(table_data))

            common_index = monthly_returns.dropna(how='all').index
            if len(common_index) < 2:
                st.warning("Not enough overlapping monthly data among these tickers.")
            else:
                st.write(f"**Usable Monthly Range:** {common_index[0].strftime('%Y-%m')} to {common_index[-1].strftime('%Y-%m')}")

                # -------------------------------------------------
                # 4.2 Fetch Fama-French 3-Factor Data
                # -------------------------------------------------
                df_factors = fetch_ff_3_factors(common_index[0], common_index[-1])
                df_factors = df_factors.loc[common_index.min():common_index.max()]

                if df_factors.empty:
                    st.warning("No Fama-French factor data available in that overlap.")
                else:
                    st.subheader("Factor Loadings (3-Factor Model)")
                    factor_cols = ["Mkt-RF", "SMB", "HML"]
                    loadings_df = pd.DataFrame(columns=["Alpha"] + factor_cols, index=valid_tickers)

                    for t in valid_tickers:
                        # y_excess = R_i - RF
                        # Align returns to factor dates
                        y_excess = monthly_returns[t].loc[df_factors.index] - df_factors["RF"]
                        X_factors = df_factors[factor_cols]

                        df_temp = pd.concat([y_excess, X_factors], axis=1).dropna()
                        if df_temp.shape[0] < 12:
                            # skip if insufficient data
                            continue
                        y_clean = df_temp.iloc[:, 0]
                        X_clean = df_temp.iloc[:, 1:]
                        X_const = sm.add_constant(X_clean)

                        model = sm.OLS(y_clean, X_const).fit()
                        loadings_df.loc[t, "Alpha"] = model.params["const"]
                        for fcol in factor_cols:
                            loadings_df.loc[t, fcol] = model.params.get(fcol, np.nan)

                    st.dataframe(loadings_df.style.format("{:.4f}"))

                    # 4.3 Compute Factor-Based Expected Returns
                    st.subheader("Factor-Based Expected Returns (Annual)")
                    avg_factor_returns = df_factors[factor_cols].mean()  # monthly
                    avg_rf = df_factors["RF"].mean()  # monthly

                    # If user wants to override Fama-French RF, we do that.
                    # We'll interpret custom_rf_annual as an annual decimal => monthly approx is custom_rf_annual / 100 / 12
                    # Actually, we've already read it as e.g. 1.5 => 1.5% annual
                    # Convert to decimal:
                    if use_custom_rf:
                        custom_rf_decimal_annual = custom_rf_annual / 100.0  # e.g. 0.015
                        custom_rf_decimal_monthly = custom_rf_decimal_annual / 12
                        used_rf_monthly = custom_rf_decimal_monthly
                    else:
                        used_rf_monthly = avg_rf  # from Fama-French

                    factor_expected_annual = {}
                    for t in valid_tickers:
                        if pd.isna(loadings_df.loc[t, "Alpha"]):
                            factor_expected_annual[t] = np.nan
                            continue
                        alpha = loadings_df.loc[t, "Alpha"]
                        betas = loadings_df.loc[t, factor_cols].values
                        factor_part = np.dot(betas, avg_factor_returns)
                        monthly_ret = alpha + factor_part + used_rf_monthly
                        annual_ret = monthly_ret * ann_factor  # multiply by 12
                        factor_expected_annual[t] = annual_ret

                    exp_ret_series = pd.Series(factor_expected_annual, name="Annual Factor-Based Return")
                    st.dataframe(exp_ret_series.to_frame().style.format("{:.2%}"))

                    # Extra Visualization: correlation of betas
                    st.subheader("Correlation of Tickers' Factor Betas")
                    loadings_only = loadings_df[factor_cols].dropna()
                    if len(loadings_only) > 1:
                        cor_loadings = loadings_only.corr()
                        fig_load, ax_load = plt.subplots(figsize=(6, 4))
                        sns.heatmap(cor_loadings, annot=True, cmap="coolwarm", center=0, ax=ax_load)
                        ax_load.set_title("Correlation of Factor Betas")
                        st.pyplot(fig_load)
                    else:
                        st.write("Not enough data for correlation of factor loadings.")

                    # -------------------------------------------------
                    # 5. OPTIMIZATION
                    # -------------------------------------------------
                    st.subheader("Factor-Based Portfolio Optimization")
                    st.write("Using the 3-Factor-based expected returns as the 'mu' vector.")
                    
                    if st.sidebar.button("Optimize Portfolio"):
                        valid_for_opt = [t for t in valid_tickers if pd.notna(factor_expected_annual[t])]

                        if not valid_for_opt:
                            st.warning("No tickers have valid factor-based expected returns.")
                        else:
                            mu_vec = np.array([factor_expected_annual[t] for t in valid_for_opt], dtype=float)

                            # We build monthly returns data just for these tickers
                            monthly_sub = monthly_returns[valid_for_opt].dropna(how='all')
                            cov_monthly = monthly_sub.cov()
                            cov_annual = cov_monthly * ann_factor

                            # Risk-free annual for Sharpe => if used_rf_monthly, multiply by 12
                            used_rf_annual = used_rf_monthly * ann_factor

                            # Constraints
                            n_assets = len(valid_for_opt)
                            bounds = [(min_weight, max_weight) for _ in range(n_assets)]
                            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
                            init_w = np.array([1/n_assets]*n_assets)

                            opt_res = minimize(
                                objective_function,
                                init_w,
                                args=(mu_vec, cov_annual, used_rf_annual, opt_objective, monthly_sub),
                                method="SLSQP",
                                bounds=bounds,
                                constraints=constraints
                            )
                            if opt_res.success:
                                opt_weights = opt_res.x
                                w_df = pd.DataFrame({"Ticker": valid_for_opt, "Weight": opt_weights})

                                st.markdown("### Optimal Portfolio Weights")
                                st.dataframe(w_df.set_index("Ticker"), use_container_width=True)

                                # Compute final metrics
                                ret_val = np.dot(opt_weights, mu_vec)
                                vol_val = np.sqrt(opt_weights.T @ cov_annual.values @ opt_weights)
                                sr_val = (ret_val - used_rf_annual) / vol_val if vol_val>0 else 0

                                # Sortino
                                port_ts = monthly_sub @ opt_weights
                                port_excess = port_ts - used_rf_monthly
                                negative_excess = port_excess[port_excess < 0]
                                if len(negative_excess)==0:
                                    so_val = float('inf')
                                else:
                                    downside_std = np.sqrt((negative_excess**2).mean()) 
                                    # approximate annualization
                                    so_val = (ret_val - used_rf_annual)/(downside_std * np.sqrt(ann_factor))

                                st.markdown("### Portfolio Metrics (Annualized)")
                                metrics_df = pd.DataFrame({
                                    "Metric": [
                                        "Expected Annual Return",
                                        "Expected Annual Volatility",
                                        "Sharpe Ratio",
                                        "Sortino Ratio"
                                    ],
                                    "Value": [
                                        f"{ret_val:.4f}",
                                        f"{vol_val:.4f}",
                                        f"{sr_val:.4f}",
                                        f"{so_val:.4f}"
                                    ]
                                })
                                st.table(metrics_df)

                                # Pie Chart
                                cmap = plt.get_cmap("tab20b")
                                colors = [cmap(i) for i in range(len(opt_weights))]

                                fig, ax = plt.subplots()
                                ax.pie(
                                    opt_weights,
                                    labels=w_df["Ticker"],
                                    autopct='%1.1f%%',
                                    startangle=90,
                                    colors=colors
                                )
                                ax.axis("equal")
                                st.markdown("### Optimal Allocation")
                                st.pyplot(fig)

                                # Download
                                csv_data = w_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Weights CSV",
                                    data=csv_data.encode('utf-8'),
                                    file_name="optimized_weights.csv",
                                    mime="text/csv",
                                )

                                # Random portfolios for frontier
                                st.markdown("### Efficient Frontier (Random Portfolios)")
                                df_rand = generate_random_portfolios(
                                    num_portfolios=2000,
                                    n_assets=n_assets,
                                    min_weight=min_weight,
                                    max_weight=max_weight,
                                    expected_returns=mu_vec,
                                    cov=cov_annual,
                                    rf=used_rf_annual,
                                    obj=opt_objective,
                                    returns_history=monthly_sub
                                )
                                if not df_rand.empty:
                                    fig_ef, ax_ef = plt.subplots()
                                    scatter = ax_ef.scatter(
                                        df_rand["Volatility"],
                                        df_rand["Return"],
                                        c=df_rand["Sharpe"],
                                        cmap="viridis",
                                        alpha=0.6
                                    )
                                    cbar = fig_ef.colorbar(scatter, ax=ax_ef)
                                    cbar.set_label("Sharpe Ratio")

                                    # Plot the optimized portfolio
                                    ax_ef.scatter(
                                        vol_val, ret_val,
                                        c="red", s=80, edgecolors="black",
                                        label="Optimized"
                                    )
                                    ax_ef.set_xlabel("Annual Volatility")
                                    ax_ef.set_ylabel("Annual Return")
                                    ax_ef.set_title("Random Portfolios & Optimized Portfolio")
                                    ax_ef.legend()
                                    st.pyplot(fig_ef)
                                else:
                                    st.warning("No random portfolios generated.")
                            else:
                                st.error("Optimization failed. Please adjust your inputs and try again.")
else:
    st.info("Select at least one ticker to enable factor-based reading and optimization.")
