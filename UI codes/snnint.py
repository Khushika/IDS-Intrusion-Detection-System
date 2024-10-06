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

    X = np.array(input_df).reshape(len(input_df), input_df.shape[1])
    return X

# Streamlit app
st.title("Intrusion Detection System")

# User input for network details
duration = st.number_input("Duration")
protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
service = st.selectbox("Service", ["http", "ftp", "smtp", "ssh", "dns", "other"])
flag = st.selectbox("Flag", ["SF", "S0", "REJ", "RSTO", "RSTR", "SH"])
src_bytes = st.number_input("Source Bytes")
dst_bytes = st.number_input("Destination Bytes")
land = st.number_input("Land")
wrong_fragment = st.number_input("Wrong Fragment")
urgent = st.number_input("Urgent")
hot = st.number_input("Hot")
num_failed_logins = st.number_input("Number of Failed Logins")
logged_in = st.number_input("Logged In")
num_compromised = st.number_input("Number of Compromised")
root_shell = st.number_input("Root Shell")
su_attempted = st.number_input("SU Attempted")
num_file_creations = st.number_input("Number of File Creations")
num_shells = st.number_input("Number of Shells")
num_access_files = st.number_input("Number of Access Files")
num_outbound_cmds = st.number_input("Number of Outbound Commands")
is_host_login = st.number_input("Is Host Login")
is_guest_login = st.number_input("Is Guest Login")
count = st.number_input("Count")
srv_count = st.number_input("Service Count")
serror_rate = st.number_input("Serror Rate")
rerror_rate = st.number_input("Rerror Rate")
same_srv_rate = st.number_input("Same Service Rate")
diff_srv_rate = st.number_input("Different Service Rate")
srv_diff_host_rate = st.number_input("Service Different Host Rate")
dst_host_count = st.number_input("Destination Host Count")
dst_host_srv_count = st.number_input("Destination Host Service Count")
dst_host_diff_srv_rate = st.number_input("Destination Host Different Service Rate")
dst_host_same_src_port_rate = st.number_input("Destination Host Same Source Port Rate")
dst_host_srv_diff_host_rate = st.number_input("Destination Host Service Different Host Rate")

# Create input data from user input
input_data = [[duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent, hot,
               num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_file_creations,
               num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, count, srv_count,
               serror_rate, rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_count,
               dst_host_srv_count, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate]]

# Add a detect button
if st.button('Detect Intrusion'):
    try:
        # Check if models are loaded
        if cnn_model is None or snn_model is None or dnn_model is None or fnn_model is None or lstm_model is None:
            st.error("Error: Models not loaded.")
        else:
            # Preprocess input data
            input_data_processed = preprocess_input(input_data)

            # Make predictions using each model
            cnn_pred = cnn_model.predict(input_data_processed.reshape(1, -1, 1))
            snn_pred = snn_model.predict(input_data_processed.reshape(1, -1, 1))
            dnn_pred = dnn_model.predict(input_data_processed)
            fnn_pred = fnn_model.predict(input_data_processed)
            lstm_pred = lstm_model.predict(input_data_processed.reshape(1, -1, 1))

            # Combine predictions using weighted averaging
            weighted_pred = (0.3 * cnn_pred + 0.2 * snn_pred + 0.2 * dnn_pred + 0.2 * fnn_pred + 0.1 * lstm_pred)

            # Display the result
            if np.round(np.mean(weighted_pred)) == 0:
                st.write('Prediction: Normal')
            else:
                st.write('Prediction: Anomaly')
    except Exception as e:
        st.error(f"Error predicting: {e}")
