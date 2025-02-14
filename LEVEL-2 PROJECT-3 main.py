import sys
sys.path.append('C:/Users/divya/OneDrive/Desktop/stock_prediction')  # Adjust if needed
import numpy as np

from stock_predictor import StockPredictor
import matplotlib.pyplot as plt
def run_prediction(symbol, start_date, end_date):
    # Initialize predictor
    print(f"Initializing predictor for {symbol}...")
    predictor = StockPredictor(symbol, start_date, end_date)  # Use the imported class name directly
    
    # Fetch data
    print("Fetching and preparing data...")
    stock_data = predictor.fetch_data()
    X, y = predictor.prepare_data(stock_data)
    
    # Split data
    print("Splitting data into train and test sets...")
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Build and train model
    print("Building and training model...")
    predictor.build_model(input_shape=(X.shape[1], X.shape[2]))
    history = predictor.train_model(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.make_predictions(X_test)
    
    # Inverse transform for actual prices
    actual_prices = predictor.scaler.inverse_transform(
        np.concatenate([X_test[:, -1, :1], np.zeros((len(X_test), 6))], axis=1)
    )[:, 0]
    
    # Plot results
    print("Plotting results...")
    predictor.plot_results(actual_prices, predictions, f"{symbol} Stock Price Prediction")
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    return predictor, history, actual_prices, predictions

if __name__ == "__main__":
    # Example configuration
    SYMBOL = 'AAPL'  # Stock symbol
    START_DATE = '2020-01-01'
    END_DATE = '2024-01-01'
    
    # Run prediction
    predictor, history, actual, predicted = run_prediction(SYMBOL, START_DATE, END_DATE)
