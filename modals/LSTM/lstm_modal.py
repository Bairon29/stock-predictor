import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def build_lstm_model(neurons, return_sequences, input_shape, dropout_rate, output_units, activation=None):
    """
    Builds a Sequential LSTM model with specified parameters.
    Args:
        neurons (int): Number of neurons in the LSTM layer.
        return_sequences (bool): Whether to return the last output in the output sequence, or the full sequence.
        input_shape (tuple): Shape of the input data (timesteps, features).
        dropout_rate (float): Dropout rate for regularization.
        output_units (int): Number of output units in the final Dense layer.
    Returns:
        model (Sequential): Compiled LSTM model.
    """
    model = Sequential([
        LSTM(neurons, return_sequences=return_sequences, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(neurons),
        Dropout(dropout_rate),
        Dense(output_units, activation=activation)
    ])
    
    return model
    
def compile_model(model, loss, optimizer, metrics):
    """
    Compiles the LSTM model with specified loss function, optimizer, and metrics.
    Args:
        model (Sequential): The LSTM model to compile.
        loss (str): Loss function to use.
        optimizer (str): Optimizer to use.
        metrics (list): List of metrics to evaluate during training and testing.
    Returns:
        model (Sequential): Compiled LSTM model.
    """
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model
    
def prepare_callback(monitor, patience, filepath):
    """
    Prepares callbacks for early stopping and model checkpointing.
    Args:
        monitor (str): Metric to monitor for early stopping and model checkpointing.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        filepath (str): Path to save the best model.
    Returns:
        callbacks (list): List of Keras callbacks for early stopping and model checkpointing.
    """
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath, monitor=monitor, save_best_only=True)
    return [early_stopping, model_checkpoint]

def train_model(model, X_train, y_train, validation_split, epochs, batch_size, callbacks):
    """
    Trains the LSTM model on the training data.
    Args:
        model (Sequential): The LSTM model to train.
        X_train (array): Training input data.
        y_train (array): Training target data.
        validation_split (float): Fraction of the training data to be used as validation data.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Number of samples per gradient update.
        callbacks (list): List of Keras callbacks for training.
    Returns:
        history: Training history object containing loss and accuracy metrics.
    """
    trained_modal = model.fit(X_train, y_train, validation_split=validation_split, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
    return trained_modal

def predict(model, X_test):
    """
    Makes predictions using the trained LSTM model.
    Args:
        model (Sequential): The trained LSTM model.
        X_test (array): Test input data.
    Returns:
        predictions (array): Predicted values from the model.
    """
    predictions = model.predict(X_test)
    return predictions

def inverse_transform_zeros_reshaped(scaler, features, data, y_price_test, last_seq_features):
    """
    Inverse transform LSTM predictions back to original scale using the last known sequence values
    for padding other features instead of zeros.
    """
    num_features = len(features)
    
    # Prepare array for predictions
    pad_data = np.zeros((len(data), num_features))
    pad_data[:, 0] = data.flatten()  # predicted Close
    if last_seq_features is not None:
        # fill other features with last known sequence values
        pad_data[:, 1:] = last_seq_features[:, -1, 1:]  # last timestep of sequence

    inverse_data = scaler.inverse_transform(pad_data)[:, 0]

    # Similarly for y_price_test
    pad_y = np.zeros((len(y_price_test), num_features))
    pad_y[:, 0] = y_price_test.flatten()
    if last_seq_features is not None:
        pad_y[:, 1:] = last_seq_features[:, -1, 1:]

    y_test_price_actual = scaler.inverse_transform(pad_y)[:, 0]

    return inverse_data, y_test_price_actual

def inverse_transform_last_seq(scaler, features, y_pred_scaled, y_test_scaled, last_seq_features):
    """
    Inverse transforms scaled Close prices using the last timestep of each sequence
    to fill the other features.
    
    Args:
        scaler (MinMaxScaler): Scaler used to scale the data.
        features (list): List of all features used in scaling.
        y_pred_scaled (np.array): Scaled predicted Close prices.
        y_test_scaled (np.array): Scaled actual Close prices for test set.
        last_seq_features (np.array): Last row of each X_test sequence (shape: [num_samples, num_features]).
        
    Returns:
        y_pred_actual, y_test_actual (np.array): Inverse-transformed Close prices.
    """
    # Prepare padded arrays
    pad_pred = np.zeros((len(y_pred_scaled), len(features)))
    pad_pred[:, 0] = y_pred_scaled.flatten()        # predicted Close
    pad_pred[:, 1:] = last_seq_features[:, 1:]     # other features from last timestep

    pad_test = np.zeros((len(y_test_scaled), len(features)))
    pad_test[:, 0] = y_test_scaled.flatten()       # actual Close
    pad_test[:, 1:] = last_seq_features[:, 1:]     # other features from last timestep

    # Inverse transform
    y_pred_actual = scaler.inverse_transform(pad_pred)[:, 0]
    y_test_actual = scaler.inverse_transform(pad_test)[:, 0]

    return y_pred_actual, y_test_actual
