import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import base64
from fpdf import FPDF
import matplotlib.pyplot as plt

st.title("Crude Flow Monitoring & Performance Analysis System")
st.markdown("**Monitoring flow rates and analyzing performance for metering & custody transfer operations**")

st.markdown("### Upload Flow Data (Excel)")
data_entry_name = st.text_input("Data Entry Name")
uploaded_file = st.file_uploader("Select an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_excel(uploaded_file)
        df["Data_Entry_Name"] = data_entry_name
        df["Pressure"] = df["Pressure"] * 14.5038  # bar to psi
        df["Temperature"] = df["Temperature"] * 9/5 + 32  # C to F
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        df = pd.DataFrame()
else:
    st.warning("Please upload a valid Excel file to proceed.")
    df = pd.DataFrame()

if not df.empty:
    st.dataframe(df.head())

    st.markdown("### Select Date Range for Analysis")
    date_range = st.date_input("Select Date Range", [df["Timestamp"].min(), df["Timestamp"].max()])
    if isinstance(date_range, list) and len(date_range) == 2:
        df = df[(df["Timestamp"] >= pd.to_datetime(date_range[0])) & (df["Timestamp"] <= pd.to_datetime(date_range[1]))]

    compare_option = st.selectbox("Compare Performance Over:", ["3 Months", "4 Months", "Semi-Annual", "Annual"])
    comparison_map = {"3 Months": "3M", "4 Months": "4M", "Semi-Annual": "6M", "Annual": "A"}
    selected_comparison = comparison_map.get(compare_option, "M")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df_numeric = df.select_dtypes(include=["number"])
    df_resampled = df_numeric.set_index(df["Timestamp"]).resample(selected_comparison).mean().reset_index()
    df = df_resampled.merge(df["Timestamp"], on="Timestamp", how="left")
    df["Data_Entry_Name"] = data_entry_name

    st.markdown("## Data Analysis Charts")
    if "Flow_Rate" in df.columns:
        fig_flow = px.line(df, x="Timestamp", y="Flow_Rate", title="Crude Oil Flow Rate Over Time (BPD)", labels={"Flow_Rate": "Flow Rate (BPD)"})
        st.plotly_chart(fig_flow)
    if "Pressure" in df.columns:
        fig_pressure = px.line(df, x="Timestamp", y="Pressure", title="Pressure Over Time (psi)", labels={"Pressure": "Pressure (psi)"})
        st.plotly_chart(fig_pressure)
    if "Temperature" in df.columns:
        fig_temp = px.line(df, x="Timestamp", y="Temperature", title="Temperature Over Time (°F)", labels={"Temperature": "Temperature (°F)"})
        st.plotly_chart(fig_temp)

    st.markdown("## Flow Rate Prediction")
    future_df = None
    if "Pressure" in df.columns and "Temperature" in df.columns:
        df_ml = df.dropna(subset=["Flow_Rate", "Pressure", "Temperature"])
        X = df_ml[["Pressure", "Temperature"]]
        y = df_ml["Flow_Rate"]
        model = LinearRegression()
        model.fit(X, y)
        coef_pressure, coef_temp = model.coef_
        intercept = model.intercept_
        X_pred = pd.DataFrame({
            "Pressure": np.linspace(X["Pressure"].min(), X["Pressure"].max(), 30),
            "Temperature": np.linspace(X["Temperature"].min(), X["Temperature"].max(), 30)
        })
        y_pred = model.predict(X_pred)
        future_df = pd.DataFrame({"Day": range(1, 31), "Predicted Flow Rate": y_pred})
        fig_pred = px.line(future_df, x="Day", y="Predicted Flow Rate", title="Predicted Flow Rate (Next 30 Days, BPD)", labels={"Predicted Flow Rate": "Predicted Flow Rate (BPD)"})
        st.plotly_chart(fig_pred)

    def export_report():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "Crude Oil Flow Analysis Report", ln=True, align='C')
        pdf.ln(10)

        if not df.empty:
            pdf.cell(200, 10, f"Data from {df['Timestamp'].min().date()} to {df['Timestamp'].max().date()}", ln=True)
            pdf.cell(200, 10, f"Avg Flow Rate: {df['Flow_Rate'].mean():.2f} BPD", ln=True)
            pdf.cell(200, 10, f"Avg Pressure: {df['Pressure'].mean():.2f} psi", ln=True)
            pdf.cell(200, 10, f"Avg Temperature: {df['Temperature'].mean():.2f} °F", ln=True)
            pdf.cell(200, 10, f"Data Entry Name: {data_entry_name}", ln=True)
            pdf.ln(5)

            def save_and_insert(fig, filename):
                fig.savefig(filename)
                pdf.image(filename, x=10, w=180)
                pdf.ln(10)

            # Flow chart
            plt.figure(figsize=(6, 4))
            plt.plot(df["Timestamp"], df["Flow_Rate"], label="Flow Rate (BPD)", color="blue")
            plt.xlabel("Timestamp")
            plt.ylabel("Flow Rate (BPD)")
            plt.title("Flow Rate Over Time (BPD)")
            plt.tight_layout()
            save_and_insert(plt, "flow.png")
            plt.close()

            # Pressure chart
            plt.figure(figsize=(6, 4))
            plt.plot(df["Timestamp"], df["Pressure"], label="Pressure (psi)", color="green")
            plt.xlabel("Timestamp")
            plt.ylabel("Pressure (psi)")
            plt.title("Pressure Over Time (psi)")
            plt.tight_layout()
            save_and_insert(plt, "pressure.png")
            plt.close()

            # Temperature chart
            plt.figure(figsize=(6, 4))
            plt.plot(df["Timestamp"], df["Temperature"], label="Temperature (°F)", color="red")
            plt.xlabel("Timestamp")
            plt.ylabel("Temperature (°F)")
            plt.title("Temperature Over Time (°F)")
            plt.tight_layout()
            save_and_insert(plt, "temperature.png")
            plt.close()

            if future_df is not None:
                plt.figure(figsize=(6, 4))
                plt.plot(future_df["Day"], future_df["Predicted Flow Rate"], label="Predicted Flow Rate (BPD)", color="purple")
                plt.xlabel("Day")
                plt.ylabel("Predicted Flow Rate (BPD)")
                plt.title("Predicted Flow Rate (Next 30 Days)")
                plt.tight_layout()
                save_and_insert(plt, "prediction.png")
                plt.close()

            pdf.set_font("Arial", style="I", size=11)
            pdf.cell(200, 10, "Note: Pressure is shown in psi, Temperature in °F, Flow Rate in BPD.", ln=True)
            pdf.ln(5)

        pdf.set_font("Arial", style="I", size=10)
        pdf.cell(200, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.output("crude_flow_report.pdf")
        return "crude_flow_report.pdf"

    if st.button("📄 Export Report as PDF"):
        file_path = export_report()
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="crude_flow_report.pdf">📥 Download PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center;'>Designed by Tareq Mageed/ Dhiqar Oil Co./Ministry of Oil</p>", unsafe_allow_html=True)
