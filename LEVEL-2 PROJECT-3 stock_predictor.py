import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping # type: ignore
class StockPredictor:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.scaler = MinMaxScaler()

    def fetch_data(self):
        """Fetch stock data and calculate technical indicators"""
        stock = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        stock['MA5'] = stock['Close'].rolling(window=5).mean()
        stock['MA20'] = stock['Close'].rolling(window=20).mean()

        delta = stock['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock['RSI'] = 100 - (100 / (1 + rs))

        exp1 = stock['Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock['Close'].ewm(span=26, adjust=False).mean()
        stock['MACD'] = exp1 - exp2

        stock['Volatility'] = stock['Close'].rolling(window=20).std()
        stock = stock.dropna()

        return stock

    def prepare_data(self, stock_data, sequence_length=60):
        """Prepare data for LSTM model"""
        features = ['Close', 'MA5', 'MA20', 'RSI', 'MACD', 'Volatility', 'Volume']
        data = stock_data[features].values
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build and compile LSTM model"""
        model = Sequential([ 
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae'])
        self.model = model  # Store the model in self.model
        return model
    def train_model(self, X, y, validation_split=0.2, epochs=50, batch_size=32):
        """Train the model with early stopping"""
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        return history

    def make_predictions(self, X_test):
        """Make predictions using the trained model"""
        predictions = self.model.predict(X_test)
        pred_prices = np.zeros((len(predictions), self.scaler.n_features_in_))
        pred_prices[:, 0] = predictions.flatten()
        return self.scaler.inverse_transform(pred_prices)[:, 0]

    def plot_results(self, actual, predicted, title="Stock Price Prediction"):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predicted, label='Predicted', color='red')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

# ✅ Correctly indented `if __name__ == "__main__":` block
if __name__ == "__main__":
    predictor = StockPredictor('AAPL', '2020-01-01', '2024-01-01')

    stock_data = predictor.fetch_data()
    X, y = predictor.prepare_data(stock_data)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
    history = predictor.train_model(X_train, y_train)

    predictions = predictor.make_predictions(X_test)
    actual_prices = predictor.scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :1], np.zeros((len(X_test), 6))], axis=1)
    )[:, 0]

    predictor.plot_results(actual_prices, predictions, "Stock Price Prediction")
