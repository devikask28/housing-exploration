import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.title("Housing Price Explorer")

# The one sentence about the model
st.write("This model estimates a house's price from two of its features. When it's wrong, a buyer could overpay or a seller could list their home too cheaply.")

# Upload + preview
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) < 3:
        st.error("This file needs at least three numeric columns.")
        st.stop()
        
    st.subheader("Preview")
    st.dataframe(df.head())
    
    # Pick a column and a chart type
    st.subheader("Chart")
    chart_type = st.radio("Chart type", ["line", "scatter", "histogram"], horizontal=True)
    column = st.selectbox("Column", numeric_cols)
    
    if chart_type == "line":
        fig = px.line(df, y=column, labels={"index": "House (ordered from most to least expensive)"}, title=f"{column} for each house")
    elif chart_type == "scatter":
        x_col = st.selectbox("Compare against", numeric_cols, index=1)
        fig = px.scatter(df, x=x_col, y=column, title=f"{column} vs {x_col}")
    else:
        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        fig.update_yaxes(title_text="Number of houses")
      
    st.plotly_chart(fig)
    
    # Two inputs + a target, fit linear regression, prediction value
    st.subheader("Prediction")
    target = st.selectbox("What to predict", numeric_cols)
    choices = [c for c in numeric_cols if c != target]
    input1 = st.selectbox("Input 1", choices)
    input2 = st.selectbox("Input 2", [c for c in choices if c != input1])
    
    data = df[[input1, input2, target]].dropna()
    model = LinearRegression()
    model.fit(data[[input1, input2]], data[target])
    
    value1 = st.number_input(input1, value=float(data[input1].mean()))
    value2 = st.number_input(input2, value=float(data[input2].mean()))
    
    # Same column names, same order as training
    new_row = pd.DataFrame([[value1, value2]], columns=[input1, input2])
    prediction = model.predict(new_row)[0]
    
    st.metric(f"Predicted {target}", f"{prediction:,.0f}")

else:
    st.info("Upload a CSV with at least three numeric columns to get started.")
