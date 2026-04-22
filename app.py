import streamlit as st
import yfinance as yf
import requests
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ------------------ UI ------------------
st.title("📈 Stock Prediction with Conformal Intervals")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
confidence = st.slider("Select Confidence Level", 0.8, 0.99, 0.9)

API_KEY = "d7ainl1r01qmvlmg4cc0d7ainl1r01qmvlmg4ccg"

# ------------------ LOAD DATA ------------------
def load_data(symbol):
    try:
        df = yf.download(symbol, period="5y", interval="1d")

        # If Yahoo fails → fallback
        if df is None or df.empty:
            raise Exception("Yahoo failed")

    except:
        st.warning("⚠️ Yahoo Finance failed, using fallback data")

        # 👉 Fallback: generate synthetic data (for demo stability)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=500)
        prices = np.cumsum(np.random.normal(0, 1, 500)) + 150

        df = pd.DataFrame({"Close": prices}, index=dates)

    df = df.dropna()

    if len(df) < 60:
        st.error("❌ Not enough data")
        return None

    return df

# ------------------ LIVE PRICE ------------------
def get_live_price(symbol):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={API_KEY}"
        data = requests.get(url).json()

        if 'c' not in data or data['c'] == 0:
            return None

        return float(data['c'])

    except:
        return None

# ------------------ RUN ------------------
if st.button("Run Prediction"):

    df = load_data(symbol)

    if df is None:
        st.stop()

    st.write("### Data (Cleaned)", df.tail())
    st.write("📊 Data shape:", df.shape)

    # ------------------ PREPROCESS ------------------
    data = df[['Close']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # ------------------ MODEL ------------------
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=0)

    # ------------------ PREDICTION ------------------
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    # ------------------ CONFORMAL ------------------
    errors = np.abs(y_test_actual - predictions)
    q = np.quantile(errors, confidence)

    lower = predictions - q
    upper = predictions + q

    # ------------------ PLOT ------------------
    st.subheader("📊 Prediction with Confidence Interval")

    plt.figure(figsize=(10,5))
    plt.plot(y_test_actual, label="Actual")
    plt.plot(predictions, label="Predicted")

    plt.fill_between(
        range(len(predictions)),
        lower.flatten(),
        upper.flatten(),
        color='gray',
        alpha=0.3,
        label="Confidence Interval"
    )

    plt.legend()
    st.pyplot(plt)

    # ------------------ LIVE DATA ------------------
    live_price = get_live_price(symbol)

    if live_price is None:
        st.warning("⚠️ Using last known price instead of live data")
        live_price = float(df['Close'].iloc[-1])

    # Smooth input
    live_price = (live_price + float(df['Close'].iloc[-1])) / 2

    # Safety check
    if len(df) < 60:
        st.error("❌ Not enough data for live prediction")
        st.stop()

    last_59 = df['Close'].values[-59:]
    new_input = np.append(last_59, live_price)

    scaled_input = scaler.transform(new_input.reshape(-1,1))
    X_live = np.array([scaled_input[:,0]]).reshape(1,60,1)

    pred = model.predict(X_live)
    pred = scaler.inverse_transform(pred)

    lower_live = pred - q
    upper_live = pred + q

    # ------------------ OUTPUT ------------------
    st.success(f"💰 Live Price: {live_price:.2f}")
    st.success(f"📈 Prediction: {pred[0][0]:.2f}")
    st.success(f"🔽 Lower Bound: {lower_live[0][0]:.2f}")
    st.success(f"🔼 Upper Bound: {upper_live[0][0]:.2f}")
    st.success(f"📊 Confidence: {confidence*100:.0f}%")

    # ------------------ CALIBRATION ------------------
    inside = (
        (y_test_actual >= (predictions - q)) &
        (y_test_actual <= (predictions + q))
    )

    coverage = np.mean(inside)

    st.subheader("📏 Calibration Check")
    st.write(f"Expected Confidence: {confidence*100:.0f}%")
    st.write(f"Actual Coverage: {coverage*100:.2f}%")

    # Calibration plot
    conf_levels = [0.8, 0.85, 0.9, 0.95]
    actual_coverage = []

    for c in conf_levels:
        q_temp = np.quantile(errors, c)

        inside_temp = (
            (y_test_actual >= (predictions - q_temp)) &
            (y_test_actual <= (predictions + q_temp))
        )

        actual_coverage.append(np.mean(inside_temp))

    plt.figure()
    plt.plot(conf_levels, actual_coverage, marker='o', label="Actual")
    plt.plot(conf_levels, conf_levels, linestyle='--', label="Ideal")
    plt.xlabel("Expected Confidence")
    plt.ylabel("Actual Coverage")
    plt.title("Calibration Plot")
    plt.legend()

    st.pyplot(plt)
