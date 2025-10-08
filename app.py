# app.py — COVID-19 Data Visualization & Storytelling Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(layout="wide", page_title="COVID-19 Data Visualization", initial_sidebar_state="expanded")

st.title("🦠 COVID-19 — Data Visualization & Storytelling Dashboard")
st.markdown("""
This dashboard visually explores the **COVID-19 pandemic dataset** to understand global and regional trends, 
distribution of cases, recovery and mortality patterns, and statistical relationships among variables.  
All plots are created using **Matplotlib** and **Seaborn**, based on the dataset `country_wise_latest.csv`.
""")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload `country_wise_latest.csv`", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("country_wise_latest.csv")
    except FileNotFoundError:
        st.error("No local CSV found. Please upload the dataset to continue.")
        st.stop()

# ----------------------------
# Data Cleaning
# ----------------------------
expected_cols = ['Country/Region','Confirmed','Deaths','Recovered','Active','New cases','New deaths','New recovered',
                 'Deaths / 100 Cases','Recovered / 100 Cases','Deaths / 100 Recovered','Confirmed last week',
                 '1 week change','1 week % increase','WHO Region']

missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.warning(f"The dataset is missing these expected columns: {missing}")

# Convert numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ----------------------------
# Global Totals
# ----------------------------
def format_number(num):
    try:
        num = int(num)
    except Exception:
        return str(num)
    if abs(num) >= 1_000_000:
        return f'{num/1_000_000:.1f}M'
    elif abs(num) >= 1000:
        return f'{num/1000:.1f}k'
    else:
        return str(num)

total_confirmed = int(df['Confirmed'].sum()) if 'Confirmed' in df.columns else 0
total_deaths = int(df['Deaths'].sum()) if 'Deaths' in df.columns else 0
total_recovered = int(df['Recovered'].sum()) if 'Recovered' in df.columns else 0
total_active = int(df['Active'].sum()) if 'Active' in df.columns else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Confirmed", format_number(total_confirmed))
k2.metric("Total Deaths", format_number(total_deaths))
k3.metric("Total Recovered", format_number(total_recovered))
k4.metric("Total Active", format_number(total_active))

st.markdown("---")

# ----------------------------
# Distribution of Cases
# ----------------------------
summary_df = pd.DataFrame({
    'Category': ['Confirmed', 'Deaths', 'Recovered', 'Active'],
    'Total Cases': [total_confirmed, total_deaths, total_recovered, total_active]
})
summary_df['Formatted'] = summary_df['Total Cases'].apply(format_number)

col1, col2 = st.columns([1,2])

with col1:
    st.subheader("📊 Global Case Distribution (Pie Chart)")
    fig1, ax1 = plt.subplots(figsize=(5,5))
    ax1.pie(summary_df['Total Cases'], labels=summary_df['Category'], autopct='%1.1f%%', startangle=60)
    ax1.axis('equal')
    st.pyplot(fig1)
    st.markdown("""
    **Explanation:**  
    The pie chart gives a proportional overview of COVID-19 categories: *Confirmed*, *Recovered*, *Deaths*, and *Active*.  
    This helps us quickly visualize the share of each outcome globally.  
    The dominance of confirmed and recovered cases indicates a significant portion of people survived the infection, 
    while deaths and active cases form a smaller segment of the overall pandemic impact.
    """)

with col2:
    st.subheader("📈 Total Cases by Category (Bar Graph)")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.barplot(x='Category', y='Total Cases', data=summary_df, ax=ax2, palette='coolwarm')
    for i, row in summary_df.iterrows():
        ax2.text(i, row['Total Cases']*1.01, row['Formatted'], ha='center')
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig2)
    st.markdown("""
    **Explanation:**  
    The bar graph presents the absolute count of each case type.  
    It shows a clear difference between categories, emphasizing that confirmed and recovered cases are far higher than deaths.  
    This kind of visualization helps policymakers identify the scale of health system workload and recovery progress globally.
    """)

st.markdown("---")

# ----------------------------
# WHO Region Analysis
# ----------------------------
if 'WHO Region' in df.columns:
    st.subheader("🌍 Cases by WHO Region")
    region_cases = df.groupby('WHO Region')[['Confirmed','Deaths','Recovered']].sum().reset_index()
    region_cases_melted = region_cases.melt(id_vars='WHO Region', var_name='Case Type', value_name='Total Cases')

    fig3, ax3 = plt.subplots(figsize=(10,5))
    sns.barplot(x='WHO Region', y='Total Cases', hue='Case Type', data=region_cases_melted, ax=ax3)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig3)
    st.markdown("""
    **Explanation:**  
    This grouped bar chart compares confirmed, recovered, and death counts across WHO regions.  
    It helps highlight regional variations — for example, some regions exhibit higher confirmed cases but also stronger recovery numbers.  
    This kind of visualization reveals how regional healthcare capacity, population density, and public health policies influenced outcomes.  
    Understanding these variations is vital for designing region-specific strategies for containment and prevention.
    """)

    # Mortality and recovery rate
    st.subheader("⚰️ Mortality and Recovery Rates by Region")
    rate_cols = [c for c in ['Deaths / 100 Cases','Recovered / 100 Cases'] if c in df.columns]
    if rate_cols:
        fig_rates, axes = plt.subplots(nrows=1, ncols=len(rate_cols), figsize=(6*len(rate_cols),4))
        if len(rate_cols) == 1:
            axes = [axes]
        for ax, rc in zip(axes, rate_cols):
            sns.barplot(x=rc, y='WHO Region', data=df.sort_values(rc, ascending=False), ax=ax)
            ax.set_title(rc)
        st.pyplot(fig_rates)
        st.markdown("""
        **Explanation:**  
        These metrics show how fatal and how recoverable COVID-19 was in different WHO regions.  
        The *Deaths / 100 Cases* metric reflects the mortality rate, while *Recovered / 100 Cases* reveals recovery effectiveness.  
        Higher recovery rates may point to efficient healthcare management and testing, whereas elevated death rates may indicate 
        regions with limited healthcare access or overwhelmed systems.
        """)
    else:
        st.info("No regional rate columns found.")

    # Regional pie chart
    region_total_cases = region_cases.copy()
    fig4, ax4 = plt.subplots(figsize=(6,6))
    ax4.pie(region_total_cases['Confirmed'], labels=region_total_cases['WHO Region'], autopct='%1.1f%%', startangle=140)
    ax4.axis('equal')
    st.pyplot(fig4)
    st.markdown("""
    **Explanation:**  
    The pie chart above shows the percentage share of confirmed cases contributed by each WHO region.  
    It highlights which regions experienced the largest portion of global infections.  
    This helps in understanding regional concentration and resource allocation priorities during the pandemic.
    """)

st.markdown("---")

# ----------------------------
# Top Countries
# ----------------------------
st.subheader("🏳️ Top Countries (Select Metric)")
metric = st.selectbox("Choose a metric to analyze", options=['Confirmed','Deaths','Recovered','Active'])
top_n = st.slider("Number of countries to display", 5, 20, 10)

if metric in df.columns:
    top_countries = df.nlargest(top_n, metric)
    fig_top, ax_top = plt.subplots(figsize=(10,6))
    sns.barplot(x=metric, y='Country/Region', data=top_countries, ax=ax_top, palette="crest")
    ax_top.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_top)
    st.markdown(f"""
    **Explanation:**  
    This visualization displays the **Top {top_n} countries** with the highest number of {metric.lower()}.  
    The graph provides an intuitive ranking that immediately identifies which nations were most severely impacted.  
    Such analysis supports international comparisons and aids in understanding global spread patterns and policy responses.
    """)
else:
    st.warning(f"Metric '{metric}' not found.")

st.markdown("---")

# ----------------------------
# Boxplot by Region
# ----------------------------
if 'WHO Region' in df.columns:
    st.subheader("📦 Distribution of Case Categories by WHO Region")
    melt_cols = [c for c in ['Confirmed','Deaths','Recovered','Active'] if c in df.columns]
    if melt_cols:
        df_melted = df.melt(id_vars='WHO Region', value_vars=melt_cols,
                             var_name='Case Type', value_name='Total Cases')
        fig_box, ax_box = plt.subplots(figsize=(12,6))
        sns.boxplot(x='WHO Region', y='Total Cases', hue='Case Type', data=df_melted, ax=ax_box)
        ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig_box)
        st.markdown("""
        **Explanation:**  
        Boxplots provide insights into data variability and outliers across WHO regions for different case types.  
        The boxes represent interquartile ranges, while whiskers show the spread of data.  
        Regions with longer boxes or extreme points show greater inconsistency in case distribution, indicating uneven reporting or outbreaks.
        """)

st.markdown("---")

# ----------------------------
# Correlation and Pairplot
# ----------------------------
st.subheader("📈 Correlation & Pairwise Relationships")
numerical_df = df.select_dtypes(include=[np.number])
if numerical_df.shape[1] >= 2:
    corr = numerical_df.corr()
    fig_heat, ax_heat = plt.subplots(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heat)
    st.pyplot(fig_heat)
    st.markdown("""
    **Explanation:**  
    The correlation heatmap shows relationships among numeric features such as confirmed, recovered, and deaths.  
    High positive correlation indicates one metric rises with another — for instance, confirmed and recovered cases.  
    Negative or weak correlations suggest limited dependency between metrics.  
    Such statistical relationships are essential for predictive modeling and epidemiological research.
    """)

st.markdown("---")

# ----------------------------
# Conclusion
# ----------------------------
st.subheader("🧾 Conclusion")
st.markdown("""
The COVID-19 data visualization provides a detailed exploration of global and regional pandemic dynamics.  
Key findings include:
- The majority of cases were concentrated in specific WHO regions, with wide differences in recovery and mortality rates.  
- Recovery numbers generally surpassed deaths, showing global medical efforts’ success despite severe regional challenges.  
- Strong positive correlation between confirmed and recovered cases indicates effective response in most regions.

This analysis highlights how data visualization simplifies the interpretation of large datasets, 
enabling better communication, awareness, and decision-making for global health policies.
""")

st.caption("📘 Dashboard created for Data Visualization Assignment | Submitted by: **Muskan Dhakrey**")
