import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit app
st.title("Interactive Statistics Learning App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Topic", [
    "Types of Variables", 
    "Types of Data", 
    "Measures of Central Tendency", 
    "Measures of Dispersion"
])

# Page 1: Types of Variables
if page == "Types of Variables":
    st.header("Types of Variables")
    st.write("""
    Variables are characteristics or attributes that can be measured or observed. 
    They are classified into three main types:
    """)
    
    st.subheader("1. Continuous Variables")
    st.write("""
    - **Definition**: Variables that can take any value within a range.
    - **Examples**: Height, weight, temperature.
    - **Visualization**: Histogram or line plot.
    """)
    st.write("**Interactive Example**")
    mean = st.slider("Mean of the distribution", 0.0, 100.0, 50.0)
    std_dev = st.slider("Standard Deviation", 0.1, 20.0, 10.0)
    continuous_data = np.random.normal(mean, std_dev, 100)
    st.write(pd.Series(continuous_data).describe())
    fig, ax = plt.subplots()
    sns.histplot(continuous_data, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("2. Categorical Variables")
    st.write("""
    - **Definition**: Variables that represent categories or groups.
    - **Examples**: Gender, color, type of car.
    - **Visualization**: Bar plot or pie chart.
    """)
    st.write("**Interactive Example**")
    categories = st.text_input("Enter categories (comma-separated)", "Red,Blue,Green")
    categories = [cat.strip() for cat in categories.split(",")]
    categorical_data = np.random.choice(categories, 100)
    st.write(pd.Series(categorical_data).value_counts())
    fig, ax = plt.subplots()
    sns.countplot(x=categorical_data, ax=ax)
    st.pyplot(fig)

    st.subheader("3. Ordinal Variables")
    st.write("""
    - **Definition**: Categorical variables with a clear ordering or ranking.
    - **Examples**: Education level (High School, Bachelor, Master, PhD), satisfaction rating (Low, Medium, High).
    - **Visualization**: Ordered bar plot.
    """)
    st.write("**Interactive Example**")
    levels = st.text_input("Enter ordinal levels (comma-separated)", "Low,Medium,High")
    levels = [level.strip() for level in levels.split(",")]
    probabilities = st.text_input("Enter probabilities for each level (comma-separated)", "0.2,0.5,0.3")
    probabilities = [float(p.strip()) for p in probabilities.split(",")]
    ordinal_data = np.random.choice(levels, 100, p=probabilities)
    st.write(pd.Series(ordinal_data).value_counts())
    fig, ax = plt.subplots()
    sns.countplot(x=ordinal_data, order=levels, ax=ax)
    st.pyplot(fig)

# Page 2: Types of Data
elif page == "Types of Data":
    st.header("Types of Data")
    st.write("""
    Data can be classified into three main types based on how it is collected:
    """)
    
    st.subheader("1. Cross-Sectional Data")
    st.write("""
    - **Definition**: Data collected at a single point in time.
    - **Examples**: Survey data, census data.
    - **Visualization**: Scatter plot or bar chart.
    """)
    st.write("**Interactive Example**")
    num_points = st.slider("Number of data points", 10, 500, 100)
    cross_sectional_data = pd.DataFrame({
        "Age": np.random.randint(18, 65, num_points),
        "Income": np.random.randint(20000, 100000, num_points)
    })
    st.write(cross_sectional_data.head())
    fig, ax = plt.subplots()
    sns.scatterplot(x="Age", y="Income", data=cross_sectional_data, ax=ax)
    st.pyplot(fig)

    st.subheader("2. Time-Series Data")
    st.write("""
    - **Definition**: Data collected over time.
    - **Examples**: Stock prices, temperature readings.
    - **Visualization**: Line plot.
    """)
    st.write("**Interactive Example**")
    start_date = st.date_input("Start date", pd.to_datetime("2023-01-01"))
    num_days = st.slider("Number of days", 10, 365, 100)
    time_series_data = pd.DataFrame({
        "Date": pd.date_range(start=start_date, periods=num_days, freq="D"),
        "Price": np.cumsum(np.random.randn(num_days)) + 100
    })
    st.write(time_series_data.head())
    fig, ax = plt.subplots()
    sns.lineplot(x="Date", y="Price", data=time_series_data, ax=ax)
    st.pyplot(fig)

    st.subheader("3. Panel Data")
    st.write("""
    - **Definition**: Data collected over time for multiple entities.
    - **Examples**: GDP of countries over years, sales data for multiple stores.
    - **Visualization**: Faceted line plot or heatmap.
    """)
    st.write("**Interactive Example**")
    num_entities = st.slider("Number of entities", 2, 10, 3)
    num_days_panel = st.slider("Number of days (panel)", 10, 365, 100)
    panel_data = pd.DataFrame({
        "Country": np.repeat([f"Country {i+1}" for i in range(num_entities)], num_days_panel),
        "Date": pd.date_range(start="2023-01-01", periods=num_days_panel, freq="D").tolist() * num_entities,
        "GDP": np.random.randn(num_entities * num_days_panel).cumsum() + 100
    })
    st.write(panel_data.head())
    fig, ax = plt.subplots()
    sns.lineplot(x="Date", y="GDP", hue="Country", data=panel_data, ax=ax)
    st.pyplot(fig)

# Page 3: Measures of Central Tendency
elif page == "Measures of Central Tendency":
    st.header("Measures of Central Tendency")
    st.write("""
    Measures of central tendency describe the center or typical value of a dataset.
    """)
    
    st.subheader("Interactive Example")
    data_input = st.text_input("Enter a list of numbers (comma-separated)", "10,20,30,40,50")
    data = [float(x.strip()) for x in data_input.split(",")]
    st.write("**Data**:", data)

    st.subheader("1. Mean")
    st.write(f"Mean: {np.mean(data):.2f}")

    st.subheader("2. Median")
    st.write(f"Median: {np.median(data):.2f}")

    st.subheader("3. Mode")
    mode_result = pd.Series(data).mode()
    if not mode_result.empty:
        st.write(f"Mode: {mode_result[0]}")
    else:
        st.write("No unique mode found.")

# Page 4: Measures of Dispersion
elif page == "Measures of Dispersion":
    st.header("Measures of Dispersion")
    st.write("""
    Measures of dispersion describe how spread out the data is.
    """)
    
    st.subheader("Interactive Example")
    data_input = st.text_input("Enter a list of numbers (comma-separated)", "10,20,30,40,50")
    data = [float(x.strip()) for x in data_input.split(",")]
    st.write("**Data**:", data)

    st.subheader("1. Variance")
    st.write(f"Variance: {np.var(data):.2f}")

    st.subheader("2. Standard Deviation")
    st.write(f"Standard Deviation: {np.std(data):.2f}")

    st.subheader("3. Covariance")
    st.write("Enter another list of numbers to calculate covariance:")
    data_input2 = st.text_input("Second list (comma-separated)", "15,25,35,45,55")
    data2 = [float(x.strip()) for x in data_input2.split(",")]
    if len(data) == len(data2):
        st.write(f"Covariance: {np.cov(data, data2)[0, 1]:.2f}")
        fig, ax = plt.subplots()
        sns.scatterplot(x=data, y=data2, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Both lists must have the same length to calculate covariance.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by José Américo")