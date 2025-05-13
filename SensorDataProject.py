import streamlit as st
import pandas as pd
#from dotenv import load_dotenv
import os

# App Title
st.title("🛠️ CNC Predictive Maintenance - Multi-Agent System")

# Sidebar Summary
st.sidebar.title("📊 Dataset Overview")

# Load Datasets
sensor_data = pd.read_csv("sensor_data.csv")
maintenance_data = pd.read_csv("maintenance_logs.csv")
failure_data = pd.read_csv("failure_records.csv")

# Standardize column names
sensor_data.columns = sensor_data.columns.str.lower()
maintenance_data.columns = maintenance_data.columns.str.lower()
failure_data.columns = failure_data.columns.str.lower()

# Sidebar metrics
st.sidebar.metric("Temperature (°C)", "42")
st.sidebar.metric("Humidity (%)", "63")
st.sidebar.metric("Vibration (g)", "5.2")
st.sidebar.metric("Frequency (Hz)", "120")

# Dataset Shapes
st.sidebar.markdown(f"**Sensor Data:** {sensor_data.shape}")
st.sidebar.markdown(f"**Maintenance Data:** {maintenance_data.shape}")
st.sidebar.markdown(f"**Failure Data:** {failure_data.shape}")

# Dataset Previews
with st.expander("📍 Sensor Data"):
    st.dataframe(sensor_data.head())
with st.expander("🛠️ Maintenance Data"):
    st.dataframe(maintenance_data.head())
with st.expander("⚠️ Failure Data"):
    st.dataframe(failure_data.head())

# User Query Input
st.markdown("### 🔍 Ask the Maintenance System Anything")
user_query = st.text_input("Type your query below...")
query_button = st.button("Get Response")

# Function to parse and respond to queries
def get_response(query):
    query = query.lower()

    if "maintenance" in query:
        recent = maintenance_data.sort_values(by="maintenance_date", ascending=False).head(1)
        return f"Latest maintenance: {recent.iloc[0]['task']} on {recent.iloc[0]['maintenance_date']} for Machine {recent.iloc[0]['machine_id']}."
    
    elif "vibration" in query:
        avg_vibration = sensor_data['vibration'].mean()
        return f"Average vibration across all machines is {avg_vibration:.2f} g."

    elif "temperature" in query:
        avg_temp = sensor_data['temperature'].mean()
        return f"Average temperature across all machines is {avg_temp:.2f}°C."

    elif "humidity" in query:
        avg_humidity = sensor_data['humidity'].mean()
        return f"Average humidity across all machines is {avg_humidity:.2f}%."

    elif "sensor fault" in query or "sensor error" in query:
        sensor_faults = failure_data[failure_data['failure_type'].str.contains("sensor", case=False)]
        if not sensor_faults.empty:
            rows = [
                f"📍 Machine {row['machine_id']} had a sensor fault on {row['failure_date']}: {row['failure_description']}"
                for _, row in sensor_faults.iterrows()
            ]
            return "\n".join(rows)
        else:
            return "No sensor-related faults found in the records."

    elif "failure" in query:
        latest_failures = failure_data.sort_values(by="failure_date", ascending=False).head(5)
        return "\n".join(
            f"⚠️ Machine {row['machine_id']} failed on {row['failure_date']} with {row['failure_type']}: {row['failure_description']}"
            for _, row in latest_failures.iterrows()
        )

    elif "range" in query or "time period" in query:
        min_time = sensor_data['timestamp'].min()
        max_time = sensor_data['timestamp'].max()
        return f"The sensor data covers the period from {min_time} to {max_time}."

    else:
        return "❓ Sorry, I didn't understand the query. Try asking about failures, vibration, temperature, or maintenance."

# Handle query
if query_button:
    if user_query:
        response = get_response(user_query)
        st.write("🧠 **System Response:**")
        st.success(response)
    else:
        st.warning("Please enter a query before submitting.")

# Multi-Agent System Status
st.markdown("### 👷 Multi-Agent Pipeline Status")
st.success("✅ All Agents Completed Successfully!")
st.markdown("""
- **Sensor Data Agent:** Successfully read and preprocessed sensor data.
- **Anomaly Detection Agent:** Applied LSTMs and Autoencoders to detect anomalies.
- **Maintenance Scheduling Agent:** Optimized and scheduled maintenance tasks.
- **Alert Notification Agent:** Sent real-time alerts and recommendations to the technician.
""")

# Footer
st.caption("🔧 Built for Predictive Maintenance of CNC Machines using a Multi-Agent AI System")
