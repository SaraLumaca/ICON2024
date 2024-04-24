# neural_network_model.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import data_visualization as DataVisualization

def train(X_train, y_train):
    
    # Standardizzazione dei dati
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Crea e addestra un modello di regressione neurale MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    # Effettua le predizioni utilizzando il modello addestrato
    return model.predict(X_test)

def evaluate(model, X_test, y_test):
    # Valuta le prestazioni del modello utilizzando MAE, MSE, RMSE e R2
    y_pred = predict(model, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

def visualize_results(model, X_test, y_test):
    predictions = predict(model, X_test)  # Calcola le predizioni utilizzando il modello
    DataVisualization.plot_neural_network_results(X_test, y_test, predictions)  # Passa le predizioni alla funzione di visualizzazione
