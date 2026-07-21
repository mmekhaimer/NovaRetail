"""
NovaRetail Customer Intelligence Dashboard
============================================
An executive-level Streamlit dashboard for analyzing customer behavior,
revenue performance, customer segments, satisfaction, retention risk,
and growth opportunities for NovaRetail.

Prepared for: Sophia Martinez, Director of Customer Intelligence
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="NovaRetail | Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "NR_dataset.xlsx"

REQUIRED_COLUMNS = [
    "CustomerID",
    "TransactionDate",
    "PurchaseAmount",
    "CustomerSatisfaction",
    "CustomerSegment",
    "ProductCategory",
    "CustomerRegion",
    "RetailChannel",
    "CustomerGender",
    "CustomerAgeGroup",
]

CATEGORICAL_COLUMNS = [
    "CustomerSegment",
    "ProductCategory",
    "CustomerAgeGroup",
    "CustomerGender",
    "CustomerRegion",
    "RetailChannel",
]

AT_RISK_KEYWORDS = ["decline", "declining", "risk", "at-risk", "churn"]
GROWTH_KEYWORDS = ["growth", "promising", "potential", "emerging"]


# =============================================================================
# DATA LOADING AND CLEANING
# =============================================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """
    Load and clean the NovaRetail transaction dataset.

    Parameters
    ----------
    path : str
        Relative path to the Excel dataset.

    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame ready for analysis, with derived time
        variables and a renamed CustomerSegment column.
    """
    df = pd.read_excel(path)

    # Strip whitespace from column names
    df.columns = [str(col).strip() for col in df.columns]

    # Rename 'label' to 'CustomerSegment' if present
    if "label" in df.columns:
        df = df.rename(columns={"label": "CustomerSegment"})

    # Convert TransactionDate to datetime, coercing invalid entries to NaT
    if "TransactionDate" in df.columns:
        df["TransactionDate"] = pd.to_datetime(
            df["TransactionDate"], errors="coerce"
        )

    # Fill missing PurchaseAmount with median
    if "PurchaseAmount" in df.columns:
        median_purchase = df["PurchaseAmount"].median()
        df["PurchaseAmount"] = df["PurchaseAmount"].fillna(median_purchase)

    # Fill missing CustomerSatisfaction with median
    if "CustomerSatisfaction" in df.columns:
        median_satisfaction = df["CustomerSatisfaction"].median()
        df["CustomerSatisfaction"] = df["CustomerSatisfaction"].fillna(
            median_satisfaction
        )

    # Clean categorical columns: fill missing values and strip whitespace
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": "Unknown", "nan": "Unknown"})

    # Drop rows missing critical identifiers
    critical_cols = [c for c in ["TransactionDate", "CustomerID"] if c in df.columns]
    if critical_cols:
        df = df.dropna(subset=critical_cols)

    # Remove duplicate transactions based on TransactionID, if present
    if "TransactionID" in df.columns:
        df = df.drop_duplicates(subset=["TransactionID"])

    # Derived time variables
    if "TransactionDate" in df.columns:
        df["YearMonth"] = df["TransactionDate"].dt.to_period("M").astype(str)
        df["Year"] = df["TransactionDate"].dt.year
        df["MonthName"] = df["TransactionDate"].dt.strftime("%B")

    return df


def validate_columns(df: pd.DataFrame) -> list:
    """
    Check that all required columns are present in the dataset.

    Returns
    -------
    list
        A list of missing required column names (empty if none missing).
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


# =============================================================================
# FILTERING LOGIC
# =============================================================================
def get_sorted_options(df: pd.DataFrame, column: str) -> list:
    """Return sorted unique non-null values for a given column."""
    if column not in df.columns:
        return []
    return sorted(df[column].dropna().unique().tolist())


def apply_filters(
    df: pd.DataFrame,
    segments: list,
    categories: list,
    regions: list,
    channels: list,
    genders: list,
    age_groups: list,
    date_range: tuple,
) -> pd.DataFrame:
    """
    Apply sidebar filter selections to the dataset.

    An empty selection for any multiselect is treated as "include all"
    for that dimension. All filters are applied simultaneously (AND logic).
    """
    filtered = df.copy()

    if segments:
        filtered = filtered[filtered["CustomerSegment"].isin(segments)]
    if categories:
        filtered = filtered[filtered["ProductCategory"].isin(categories)]
    if regions:
        filtered = filtered[filtered["CustomerRegion"].isin(regions)]
    if channels:
        filtered = filtered[filtered["RetailChannel"].isin(channels)]
    if genders:
        filtered = filtered[filtered["CustomerGender"].isin(genders)]
    if age_groups:
        filtered = filtered[filtered["CustomerAgeGroup"].isin(age_groups)]

    if date_range and len(date_range) == 2 and "TransactionDate" in filtered.columns:
        start_date, end_date = date_range
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[
            (filtered["TransactionDate"] >= start_ts)
            & (filtered["TransactionDate"] <= end_ts)
        ]

    return filtered


# =============================================================================
# KPI CALCULATIONS
# =============================================================================
def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Compute executive KPI values from the filtered dataset.

    Returns
    -------
    dict
        Dictionary containing total revenue, unique customers,
        number of transactions, average purchase amount, and
        average customer satisfaction.
    """
    if df.empty:
        return {
            "total_revenue": 0.0,
            "unique_customers": 0,
            "num_transactions": 0,
            "avg_purchase": 0.0,
            "avg_satisfaction": 0.0,
        }

    total_revenue = df["PurchaseAmount"].sum()
    unique_customers = df["CustomerID"].nunique()

    if "TransactionID" in df.columns:
        num_transactions = df["TransactionID"].nunique()
    else:
        num_transactions = len(df)

    avg_purchase = df["PurchaseAmount"].mean()
    avg_satisfaction = df["CustomerSatisfaction"].mean()

    return {
        "total_revenue": total_revenue,
        "unique_customers": unique_customers,
        "num_transactions": num_transactions,
        "avg_purchase": avg_purchase,
        "avg_satisfaction": avg_satisfaction,
    }


# =============================================================================
# CHART BUILDERS
# =============================================================================
def build_revenue_by_segment_chart(df: pd.DataFrame):
    """Vertical bar chart of total revenue by customer segment."""
    revenue_by_segment = (
        df.groupby("CustomerSegment", as_index=False)["PurchaseAmount"]
        .sum()
        .sort_values("PurchaseAmount", ascending=False)
    )
    revenue_by_segment["RevenueLabel"] = revenue_by_segment["PurchaseAmount"].apply(
        lambda x: f"${x:,.0f}"
    )

    fig = px.bar(
        revenue_by_segment,
        x="CustomerSegment",
        y="PurchaseAmount",
        text="RevenueLabel",
        template="plotly_white",
        title="Revenue by Customer Segment",
        labels={"PurchaseAmount": "Total Revenue", "CustomerSegment": "Customer Segment"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def build_customer_distribution_chart(df: pd.DataFrame):
    """Donut chart of unique customer counts by segment."""
    unique_customers = df.drop_duplicates(subset=["CustomerID"])
    distribution = (
        unique_customers.groupby("CustomerSegment", as_index=False)["CustomerID"]
        .count()
        .rename(columns={"CustomerID": "CustomerCount"})
    )

    fig = px.pie(
        distribution,
        names="CustomerSegment",
        values="CustomerCount",
        hole=0.45,
        template="plotly_white",
        title="Distribution of Customers Across Segments",
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} customers (%{percent})",
    )
    return fig


def build_revenue_by_category_chart(df: pd.DataFrame):
    """Horizontal bar chart of total revenue by product category."""
    revenue_by_category = (
        df.groupby("ProductCategory", as_index=False)["PurchaseAmount"]
        .sum()
        .sort_values("PurchaseAmount", ascending=True)
    )
    revenue_by_category["RevenueLabel"] = revenue_by_category["PurchaseAmount"].apply(
        lambda x: f"${x:,.0f}"
    )

    fig = px.bar(
        revenue_by_category,
        y="ProductCategory",
        x="PurchaseAmount",
        text="RevenueLabel",
        orientation="h",
        template="plotly_white",
        title="Revenue by Product Category",
        labels={"PurchaseAmount": "Total Revenue", "ProductCategory": "Product Category"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def build_revenue_by_region_chart(df: pd.DataFrame):
    """Vertical bar chart of total revenue by region."""
    revenue_by_region = (
        df.groupby("CustomerRegion", as_index=False)["PurchaseAmount"]
        .sum()
        .sort_values("PurchaseAmount", ascending=False)
    )
    revenue_by_region["RevenueLabel"] = revenue_by_region["PurchaseAmount"].apply(
        lambda x: f"${x:,.0f}"
    )

    fig = px.bar(
        revenue_by_region,
        x="CustomerRegion",
        y="PurchaseAmount",
        text="RevenueLabel",
        template="plotly_white",
        title="Revenue by Region",
        labels={"PurchaseAmount": "Total Revenue", "CustomerRegion": "Region"},
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="%{x}: $%{y:,.2f}<extra></extra>",
    )
    fig.update_layout(showlegend=False)
    return fig


def build_revenue_by_channel_chart(df: pd.DataFrame):
    """Donut chart of revenue share by retail channel."""
    revenue_by_channel = df.groupby("RetailChannel", as_index=False)["PurchaseAmount"].sum()

    fig = px.pie(
        revenue_by_channel,
        names="RetailChannel",
        values="PurchaseAmount",
        hole=0.45,
        template="plotly_white",
        title="Revenue Share by Retail Channel",
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}: $%{value:,.2f} (%{percent})",
    )
    return fig


def build_satisfaction_by_segment_chart(df: pd.DataFrame):
    """Vertical bar chart of average satisfaction by segment."""
    satisfaction_by_segment = (
        df.groupby("CustomerSegment", as_index=False)["CustomerSatisfaction"]
        .mean()
        .sort_values("CustomerSatisfaction", ascending=False)
    )
    satisfaction_by_segment["SatisfactionLabel"] = satisfaction_by_segment[
        "CustomerSatisfaction"
    ].apply(lambda x: f"{x:.2f}")

    fig = px.bar(
        satisfaction_by_segment,
        x="CustomerSegment",
        y="CustomerSatisfaction",
        text="SatisfactionLabel",
        template="plotly_white",
        title="Average Customer Satisfaction by Segment",
        labels={
            "CustomerSatisfaction": "Average Satisfaction",
            "CustomerSegment": "Customer Segment",
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    return fig


def build_revenue_trend_chart(df: pd.DataFrame, granularity: str):
    """
    Line chart of total revenue over time at the selected granularity.

    Parameters
    ----------
    df : pd.DataFrame
        The filtered dataset.
    granularity : str
        One of "Day", "Month", "Year".
    """
    freq_map = {"Day": "D", "Month": "M", "Year": "Y"}
    freq = freq_map.get(granularity, "D")

    trend_df = df.copy()
    trend_df["TimeBucket"] = trend_df["TransactionDate"].dt.to_period(freq).dt.to_timestamp()

    trend_series = (
        trend_df.groupby("TimeBucket", as_index=False)["PurchaseAmount"]
        .sum()
        .sort_values("TimeBucket")
    )

    fig = px.line(
        trend_series,
        x="TimeBucket",
        y="PurchaseAmount",
        markers=True,
        template="plotly_white",
        title=f"Revenue Trend by {granularity}",
        labels={"TimeBucket": "Date", "PurchaseAmount": "Total Revenue"},
    )
    fig.update_traces(hovertemplate="%{x}: $%{y:,.2f}<extra></extra>")
    fig.update_layout(hovermode="x unified")
    return fig


def build_top_customers_chart(top_customers_df: pd.DataFrame):
    """Horizontal bar chart of top 10 customers by revenue."""
    fig = px.bar(
        top_customers_df.sort_values("TotalRevenue", ascending=True),
        y="CustomerID",
        x="TotalRevenue",
        orientation="h",
        template="plotly_white",
        title="Top 10 Customers by Revenue",
        labels={"TotalRevenue": "Total Revenue", "CustomerID": "Customer ID"},
    )
    fig.update_traces(hovertemplate="Customer %{y}: $%{x:,.2f}<extra></extra>")
    fig.update_layout(showlegend=False)
    return fig


# =============================================================================
# AUTOMATED EXECUTIVE INSIGHTS
# =============================================================================
def generate_insights(df: pd.DataFrame) -> dict:
    """
    Generate plain-language executive insights based on the currently
    filtered dataset.

    Returns
    -------
    dict
        Dictionary of insight strings keyed by insight name.
    """
    insights = {}

    total_revenue = df["PurchaseAmount"].sum()

    revenue_by_segment = df.groupby("CustomerSegment")["PurchaseAmount"].sum().sort_values(
        ascending=False
    )

    # A. Highest-Revenue Segment
    if not revenue_by_segment.empty:
        top_segment = revenue_by_segment.index[0]
        top_segment_revenue = revenue_by_segment.iloc[0]
        top_segment_share = (
            (top_segment_revenue / total_revenue * 100) if total_revenue > 0 else 0
        )
        insights["revenue_leader"] = (
            f"**{top_segment}** is the highest-revenue segment, generating "
            f"**${top_segment_revenue:,.0f}** and accounting for "
            f"**{top_segment_share:.1f}%** of total revenue."
        )
    else:
        insights["revenue_leader"] = "No segment data available for the current filters."

    # B. Lowest-Revenue Segment
    if len(revenue_by_segment) > 1:
        bottom_segment = revenue_by_segment.index[-1]
        bottom_segment_revenue = revenue_by_segment.iloc[-1]
        bottom_segment_share = (
            (bottom_segment_revenue / total_revenue * 100) if total_revenue > 0 else 0
        )
        insights["underperforming_segment"] = (
            f"**{bottom_segment}** is the lowest-revenue segment, generating only "
            f"**${bottom_segment_revenue:,.0f}** (**{bottom_segment_share:.1f}%** of total "
            f"revenue). This segment may warrant further investigation."
        )
    else:
        insights["underperforming_segment"] = (
            "Not enough segment variety in the current filters to identify an "
            "underperforming segment."
        )

    # C. Strongest Region
    revenue_by_region = df.groupby("CustomerRegion")["PurchaseAmount"].sum().sort_values(
        ascending=False
    )
    if not revenue_by_region.empty:
        top_region = revenue_by_region.index[0]
        top_region_revenue = revenue_by_region.iloc[0]
        insights["regional_strength"] = (
            f"**{top_region}** is the strongest region, generating "
            f"**${top_region_revenue:,.0f}** in revenue."
        )
    else:
        insights["regional_strength"] = "No regional data available for the current filters."

    # D. At-Risk Segment
    segment_names = df["CustomerSegment"].astype(str).unique().tolist()
    at_risk_matches = [
        seg for seg in segment_names
        if any(keyword in seg.lower() for keyword in AT_RISK_KEYWORDS)
    ]

    if at_risk_matches:
        at_risk_segment = at_risk_matches[0]
        subset = df[df["CustomerSegment"] == at_risk_segment]
        at_risk_customers = subset["CustomerID"].nunique()
        at_risk_satisfaction = subset["CustomerSatisfaction"].mean()
        at_risk_revenue = subset["PurchaseAmount"].sum()
        insights["warning_signs"] = (
            f"**{at_risk_segment}** is a flagged at-risk segment with "
            f"**{at_risk_customers:,}** unique customers, average satisfaction of "
            f"**{at_risk_satisfaction:.2f}**, and **${at_risk_revenue:,.0f}** in revenue. "
            f"This segment may warrant proactive retention efforts."
        )
    else:
        satisfaction_by_segment = df.groupby("CustomerSegment")["CustomerSatisfaction"].mean()
        if not satisfaction_by_segment.empty:
            lowest_satisfaction_segment = satisfaction_by_segment.idxmin()
            lowest_satisfaction_value = satisfaction_by_segment.min()
            insights["warning_signs"] = (
                f"No explicit at-risk segment was found. **{lowest_satisfaction_segment}** "
                f"has the lowest average satisfaction score (**{lowest_satisfaction_value:.2f}**) "
                f"and may require closer monitoring."
            )
        else:
            insights["warning_signs"] = "No satisfaction data available for the current filters."

    # E. Growth Opportunity
    growth_matches = [
        seg for seg in segment_names
        if any(keyword in seg.lower() for keyword in GROWTH_KEYWORDS)
    ]

    if growth_matches:
        growth_segment = growth_matches[0]
        subset = df[df["CustomerSegment"] == growth_segment]
        growth_revenue = subset["PurchaseAmount"].sum()
        growth_share = (growth_revenue / total_revenue * 100) if total_revenue > 0 else 0
        growth_customers = subset["CustomerID"].nunique()
        growth_satisfaction = subset["CustomerSatisfaction"].mean()
        insights["growth_opportunity"] = (
            f"**{growth_segment}** represents a growth opportunity, contributing "
            f"**{growth_share:.1f}%** of revenue across **{growth_customers:,}** customers "
            f"with an average satisfaction of **{growth_satisfaction:.2f}**. This segment "
            f"may respond well to upselling or conversion campaigns."
        )
    else:
        if not revenue_by_segment.empty:
            median_revenue = revenue_by_segment.median()
            satisfaction_by_segment = df.groupby("CustomerSegment")["CustomerSatisfaction"].mean()
            below_median = revenue_by_segment[revenue_by_segment < median_revenue].index.tolist()
            if below_median:
                candidate_satisfaction = satisfaction_by_segment.loc[below_median].sort_values(
                    ascending=False
                )
                candidate_segment = candidate_satisfaction.index[0]
                candidate_value = candidate_satisfaction.iloc[0]
                insights["growth_opportunity"] = (
                    f"No explicit growth segment was found. **{candidate_segment}** shows "
                    f"relatively strong satisfaction (**{candidate_value:.2f}**) despite "
                    f"below-median revenue, suggesting upside potential."
                )
            else:
                insights["growth_opportunity"] = (
                    "No clear below-median segment was identified for growth targeting."
                )
        else:
            insights["growth_opportunity"] = "No segment data available for the current filters."

    # F. Marketing Investment Recommendation
    revenue_by_channel = df.groupby("RetailChannel")["PurchaseAmount"].sum().sort_values(
        ascending=False
    )
    revenue_by_category = df.groupby("ProductCategory")["PurchaseAmount"].sum().sort_values(
        ascending=False
    )

    if (
        not revenue_by_channel.empty
        and not revenue_by_category.empty
        and not revenue_by_segment.empty
    ):
        top_channel = revenue_by_channel.index[0]
        top_category = revenue_by_category.index[0]
        top_seg_for_rec = revenue_by_segment.index[0]
        insights["recommendation"] = (
            f"NovaRetail should prioritize marketing investment through the "
            f"**{top_channel}** channel, promoting **{top_category}** products to the "
            f"**{top_seg_for_rec}** segment. Pairing this with loyalty incentives, "
            f"cross-selling campaigns, and targeted retention offers could help "
            f"maximize customer lifetime value across the highest-performing combination "
            f"of channel, category, and segment."
        )
    else:
        insights["recommendation"] = (
            "Insufficient data across channel, category, and segment to generate a "
            "specific recommendation."
        )

    # G. Revenue Concentration
    if not revenue_by_segment.empty and total_revenue > 0:
        top_share = revenue_by_segment.iloc[0] / total_revenue * 100
        top_three_share = revenue_by_segment.iloc[:3].sum() / total_revenue * 100
        if top_share > 40:
            insights["concentration_warning"] = (
                f"Revenue is highly concentrated: the leading segment alone accounts for "
                f"**{top_share:.1f}%** of total revenue, and the top three segments "
                f"together represent **{top_three_share:.1f}%**. This concentration "
                f"increases exposure if the leading segment weakens."
            )
        else:
            insights["concentration_warning"] = (
                f"Revenue is reasonably diversified: the leading segment accounts for "
                f"**{top_share:.1f}%** of total revenue, with the top three segments "
                f"representing **{top_three_share:.1f}%** combined."
            )
    else:
        insights["concentration_warning"] = "No revenue concentration data available."

    # H. Trend Direction
    monthly_revenue = (
        df.groupby(df["TransactionDate"].dt.to_period("M").astype(str))["PurchaseAmount"]
        .sum()
        .sort_index()
    )

    if len(monthly_revenue) >= 2:
        latest_month_revenue = monthly_revenue.iloc[-1]
        previous_month_revenue = monthly_revenue.iloc[-2]
        latest_month_label = monthly_revenue.index[-1]
        previous_month_label = monthly_revenue.index[-2]

        if previous_month_revenue > 0:
            pct_change = (
                (latest_month_revenue - previous_month_revenue) / previous_month_revenue * 100
            )
        else:
            pct_change = 0.0

        if pct_change > 1:
            direction = "improving"
        elif pct_change < -1:
            direction = "declining"
        else:
            direction = "stable"

        insights["trend_direction"] = (
            f"Revenue trend is **{direction}**: **{latest_month_label}** generated "
            f"**${latest_month_revenue:,.0f}** compared to **${previous_month_revenue:,.0f}** "
            f"in **{previous_month_label}**, a change of **{pct_change:+.1f}%**."
        )
    else:
        insights["trend_direction"] = (
            "Not enough monthly data available to determine a trend direction."
        )

    return insights


def compute_top_customers(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Compute a summary table of the top N customers by total revenue.

    Returns
    -------
    pd.DataFrame
        Table with total revenue, transaction count, average purchase,
        average satisfaction, and most frequent product category.
    """
    if df.empty:
        return pd.DataFrame()

    agg_dict = {
        "PurchaseAmount": ["sum", "mean"],
        "CustomerSatisfaction": "mean",
    }

    if "TransactionID" in df.columns:
        transaction_counts = df.groupby("CustomerID")["TransactionID"].nunique()
    else:
        transaction_counts = df.groupby("CustomerID").size()

    grouped = df.groupby("CustomerID").agg(agg_dict)
    grouped.columns = ["TotalRevenue", "AvgPurchaseAmount", "AvgSatisfaction"]
    grouped["NumTransactions"] = transaction_counts
    grouped = grouped.reset_index()

    if "ProductCategory" in df.columns:
        top_category = (
            df.groupby("CustomerID")["ProductCategory"]
            .agg(lambda x: x.value_counts().idxmax() if len(x) > 0 else "Unknown")
            .reset_index()
            .rename(columns={"ProductCategory": "TopProductCategory"})
        )
        grouped = grouped.merge(top_category, on="CustomerID", how="left")

    grouped = grouped.sort_values("TotalRevenue", ascending=False).head(top_n)
    return grouped


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Run the NovaRetail Customer Intelligence Dashboard."""

    st.title("NovaRetail Customer Intelligence Dashboard")
    st.markdown(
        "Prepared for **Sophia Martinez, Director of Customer Intelligence**. "
        "This dashboard provides insights into **revenue drivers**, "
        "**at-risk customer segments**, **customer satisfaction**, "
        "**growth opportunities**, and **retention priorities**."
    )
    st.divider()

    # --- Load data with error handling ---
    try:
        raw_df = load_data(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Could not find `{DATA_PATH}`. Please make sure `NR_dataset.xlsx` "
            "is placed in the same folder as `app.py`."
        )
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while reading the data file: {e}")
        st.stop()

    # --- Validate required columns ---
    missing_columns = validate_columns(raw_df)
    if missing_columns:
        st.error(
            "The dataset is missing required columns needed to build this dashboard:"
        )
        st.write(missing_columns)
        st.stop()

    # =========================================================================
    # SIDEBAR FILTERS
    # =========================================================================
    st.sidebar.header("Filters")

    if st.sidebar.button("Reset Filters"):
        for key in [
            "segment_filter",
            "category_filter",
            "region_filter",
            "channel_filter",
            "gender_filter",
            "age_group_filter",
            "date_range_filter",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    segment_options = get_sorted_options(raw_df, "CustomerSegment")
    category_options = get_sorted_options(raw_df, "ProductCategory")
    region_options = get_sorted_options(raw_df, "CustomerRegion")
    channel_options = get_sorted_options(raw_df, "RetailChannel")
    gender_options = get_sorted_options(raw_df, "CustomerGender")
    age_group_options = get_sorted_options(raw_df, "CustomerAgeGroup")

    selected_segments = st.sidebar.multiselect(
        "Customer Segment", options=segment_options, key="segment_filter"
    )
    selected_categories = st.sidebar.multiselect(
        "Product Category", options=category_options, key="category_filter"
    )
    selected_regions = st.sidebar.multiselect(
        "Region", options=region_options, key="region_filter"
    )
    selected_channels = st.sidebar.multiselect(
        "Retail Channel", options=channel_options, key="channel_filter"
    )
    selected_genders = st.sidebar.multiselect(
        "Gender", options=gender_options, key="gender_filter"
    )
    selected_age_groups = st.sidebar.multiselect(
        "Age Group", options=age_group_options, key="age_group_filter"
    )

    min_date = raw_df["TransactionDate"].min().date()
    max_date = raw_df["TransactionDate"].max().date()

    selected_date_range = st.sidebar.date_input(
        "Transaction Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range_filter",
    )

    if isinstance(selected_date_range, tuple) and len(selected_date_range) == 2:
        date_range = selected_date_range
    else:
        date_range = (min_date, max_date)

    filtered_df = apply_filters(
        raw_df,
        selected_segments,
        selected_categories,
        selected_regions,
        selected_channels,
        selected_genders,
        selected_age_groups,
        date_range,
    )

    if filtered_df.empty:
        st.warning(
            "No data matches the current filter selection. Please broaden your filters."
        )
        st.stop()

    # =========================================================================
    # KPI CARDS
    # =========================================================================
    kpis = compute_kpis(filtered_df)
    unfiltered_kpis = compute_kpis(raw_df)

    kpi_cols = st.columns(5)

    revenue_delta = kpis["total_revenue"] - unfiltered_kpis["total_revenue"]
    customers_delta = kpis["unique_customers"] - unfiltered_kpis["unique_customers"]
    transactions_delta = kpis["num_transactions"] - unfiltered_kpis["num_transactions"]
    purchase_delta = kpis["avg_purchase"] - unfiltered_kpis["avg_purchase"]
    satisfaction_delta = kpis["avg_satisfaction"] - unfiltered_kpis["avg_satisfaction"]

    with kpi_cols[0]:
        st.metric(
            "Total Revenue",
            f"${kpis['total_revenue']:,.0f}",
            delta=f"${revenue_delta:,.0f}",
        )
    with kpi_cols[1]:
        st.metric(
            "Unique Customers",
            f"{kpis['unique_customers']:,}",
            delta=f"{customers_delta:,}",
        )
    with kpi_cols[2]:
        st.metric(
            "Transactions",
            f"{kpis['num_transactions']:,}",
            delta=f"{transactions_delta:,}",
        )
    with kpi_cols[3]:
        st.metric(
            "Avg Purchase Amount",
            f"${kpis['avg_purchase']:,.2f}",
            delta=f"${purchase_delta:,.2f}",
        )
    with kpi_cols[4]:
        st.metric(
            "Avg Satisfaction",
            f"{kpis['avg_satisfaction']:.2f}",
            delta=f"{satisfaction_delta:.2f}",
        )

    st.divider()

    # =========================================================================
    # REVENUE AND SEGMENT ANALYSIS
    # =========================================================================
    st.subheader("Revenue and Segment Analysis")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.plotly_chart(
            build_revenue_by_segment_chart(filtered_df), use_container_width=True
        )
    with row1_col2:
        st.plotly_chart(
            build_customer_distribution_chart(filtered_df), use_container_width=True
        )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.plotly_chart(
            build_revenue_by_category_chart(filtered_df), use_container_width=True
        )
    with row2_col2:
        st.plotly_chart(
            build_revenue_by_region_chart(filtered_df), use_container_width=True
        )

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.plotly_chart(
            build_revenue_by_channel_chart(filtered_df), use_container_width=True
        )
    with row3_col2:
        st.plotly_chart(
            build_satisfaction_by_segment_chart(filtered_df), use_container_width=True
        )

    st.divider()

    # =========================================================================
    # REVENUE TREND
    # =========================================================================
    st.subheader("Revenue Trend")

    granularity = st.radio(
        "Select Time Granularity", options=["Day", "Month", "Year"], index=0, horizontal=True
    )

    st.plotly_chart(
        build_revenue_trend_chart(filtered_df, granularity), use_container_width=True
    )

    st.divider()

    # =========================================================================
    # EXECUTIVE SUMMARY AND INSIGHTS
    # =========================================================================
    st.subheader("Executive Summary & Insights")

    insights = generate_insights(filtered_df)

    st.markdown("**Revenue Leader**")
    st.success(insights["revenue_leader"])

    st.markdown("**Underperforming Segment**")
    st.warning(insights["underperforming_segment"])

    st.markdown("**Regional Strength**")
    st.info(insights["regional_strength"])

    st.markdown("**Warning Signs**")
    st.warning(insights["warning_signs"])

    st.markdown("**Growth Opportunity**")
    st.info(insights["growth_opportunity"])

    st.markdown("**Revenue Concentration**")
    st.warning(insights["concentration_warning"])

    st.markdown("**Trend Direction**")
    st.info(insights["trend_direction"])

    st.markdown("**Recommendation**")
    st.success(insights["recommendation"])

    st.divider()

    # =========================================================================
    # TOP CUSTOMERS (OPTIONAL CUSTOMER-LEVEL ANALYSIS)
    # =========================================================================
    st.subheader("Top 10 Customers by Revenue")

    top_customers_df = compute_top_customers(filtered_df, top_n=10)

    if not top_customers_df.empty:
        st.plotly_chart(
            build_top_customers_chart(top_customers_df), use_container_width=True
        )

        display_df = top_customers_df.copy()
        display_df["TotalRevenue"] = display_df["TotalRevenue"].apply(lambda x: f"${x:,.2f}")
        display_df["AvgPurchaseAmount"] = display_df["AvgPurchaseAmount"].apply(
            lambda x: f"${x:,.2f}"
        )
        display_df["AvgSatisfaction"] = display_df["AvgSatisfaction"].apply(
            lambda x: f"{x:.2f}"
        )
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No customer-level data available for the current filters.")

    st.divider()

    # =========================================================================
    # DATA PREVIEW AND DOWNLOAD
    # =========================================================================
    with st.expander("View Filtered Data"):
        st.write(f"Number of filtered rows: **{len(filtered_df):,}**")
        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_data,
            file_name="novaretail_filtered_data.csv",
            mime="text/csv",
        )

    st.divider()

    # =========================================================================
    # FOOTER
    # =========================================================================
    generation_time = datetime.now().strftime("%B %d, %Y %I:%M %p")
    data_min_date = raw_df["TransactionDate"].min().strftime("%B %d, %Y")
    data_max_date = raw_df["TransactionDate"].max().strftime("%B %d, %Y")

    st.markdown(
        f"""
        ---
        **Dashboard generated:** {generation_time}
        **Data range:** {data_min_date} to {data_max_date}
        **Filtered observations:** {len(filtered_df):,}
        """
    )


# =============================================================================
# DEPLOYMENT NOTES
# =============================================================================
# 1. Place `app.py`, `NR_dataset.xlsx`, and `requirements.txt` in the same
#    project folder.
# 2. To run locally:
#       streamlit run app.py
# 3. To deploy on Streamlit Community Cloud:
#       - Push the project folder to a GitHub repository.
#       - Sign in to https://share.streamlit.io with your GitHub account.
#       - Create a new app, select the repository and branch, and set
#         `app.py` as the entry point.
#       - Streamlit Cloud will automatically install packages listed in
#         `requirements.txt`.
# =============================================================================

if __name__ == "__main__":
    main()
