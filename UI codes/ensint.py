import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load the trained models
try:
    cnn_model = load_model("cnn_model.h5")
    snn_model = load_model("snn_model.h5")
    dnn_model = load_model("dnn_model.h5")
    fnn_model = load_model("fnn_model.h5")
    lstm_model = load_model("lstm_model.h5")
except Exception as e:
    st.error(f"Error loading models: {e}")

# Define function for preprocessing input data
def preprocess_input(input_data):
    input_df = pd.DataFrame(input_data, columns=[
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_file_creations', 'num_shells', 'num_access_files',
        'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
        'srv_count', 'serror_rate', 'rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate'])

    categorical_columns = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        input_df[col] = label_encoders[col].fit_transform(input_df[col])

    X = np.array(input_df)
    return X

# Streamlit app
st.title("Multiphase Intrusion Detection System using Machine Learning")

# Create 11 columns
cols = st.columns(3)

# User input for network details
with cols[0]:
    duration = st.number_input("Duration", key="duration")
    src_bytes = st.number_input("Source Bytes", key="source_bytes")
    num_failed_logins = st.number_input("Number of Failed Logins", key="failed_logins")
    root_shell = st.number_input("Root Shell", key="root_shell")
    is_host_login = st.number_input("Is Host Login", key="host_login")
    srv_count = st.number_input("Service Count", key="service_count")
    rerror_rate = st.number_input("Rerror Rate", key="rerror_rate")
    same_srv_rate = st.number_input("Same Service Rate", key="same_srv_rate")
    dst_host_count = st.number_input("Destination Host Count", key="host_count")
    protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], key="protocol_type")
    dst_bytes = st.number_input("Destination Bytes", key="destination_bytes")

with cols[1]:
    logged_in = st.number_input("Logged In", key="logged_in")
    num_compromised = st.number_input("Number of Compromised", key="num_compromised")
    su_attempted = st.number_input("SU Attempted", key="su_attempted")
    is_guest_login = st.number_input("Is Guest Login", key="guest_login")
    serror_rate = st.number_input("Serror Rate", key="serror_rate")
    diff_srv_rate = st.number_input("Different Service Rate", key="diff_srv_rate")
    dst_host_srv_count = st.number_input("Destination Host Service Count", key="host_service_count")
    service = st.selectbox("Service", ["http", "ftp", "smtp", "ssh", "dns", "other"], key="service")
    flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR", "SH"], key="flag")
    land = st.number_input("Land", key="land")
    hot = st.number_input("Hot", key="hot")

with cols[2]:
    wrong_fragment = st.number_input("Wrong Fragment", key="wrong_fragment")
    urgent = st.number_input("Urgent", key="urgent")
    num_file_creations = st.number_input("Number of File Creations", key="file_creations")
    num_shells = st.number_input("Number of Shells", key="shells")
    num_access_files = st.number_input("Number of Access Files", key="access_files")
    num_outbound_cmds = st.number_input("Number of Outbound Commands", key="outbound_cmds")
    count = st.number_input("Count", key="count")
    srv_diff_host_rate = st.number_input("Service Different Host Rate", key="service_host_rate")
    dst_host_diff_srv_rate = st.number_input("Destination Host Different Service Rate", key="host_diff_service_rate")
    dst_host_same_src_port_rate = st.number_input("Destination Host Same Source Port Rate", key="same_source_port_rate")
    dst_host_srv_diff_host_rate = st.number_input("Destination Host Service Different Host Rate", key="service_diff_host_rate")

# Create input data from user input
input_data = [[duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot,
               num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_file_creations,
               num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count,
               serror_rate, rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count,
               dst_host_srv_count, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate]]

# Add a detect button
if st.button('Detect Intrusion'):
    try:
        # Preprocess input data
        input_data_processed = preprocess_input(input_data)

        # Make predictions using each model
        cnn_pred = cnn_model.predict(input_data_processed)
        snn_pred = snn_model.predict(input_data_processed)
        dnn_pred = dnn_model.predict(input_data_processed)
        fnn_pred = fnn_model.predict(input_data_processed)
        lstm_pred = lstm_model.predict(input_data_processed.reshape(1, -1, 33))

        # Combine predictions using weighted averaging
        weighted_pred = (0.3 * cnn_pred + 0.2 * snn_pred + 0.2 * dnn_pred + 0.2 * fnn_pred + 0.1 * lstm_pred)

        # Set threshold for anomaly detection
        anomaly_threshold = 0.5  # Adjust as needed

        # Display the output
        if np.mean(weighted_pred) >= anomaly_threshold:
            st.write('<div style="padding:10px; border-radius:5px;"><h2>Prediction: Anomaly</h2></div>', unsafe_allow_html=True)
        else:
            st.write('<div style="padding:10px; border-radius:5px;"><h2>Prediction: Normal</h2></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error predicting: {e}")

