import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn import preprocessing, metrics
import data_visualization as DataVisualization
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

def prepare_data_for_modeling(df, features, target):
    X = df[features]
    y = df[target]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_linear_regression_model(X_train, y_train):
    # Inizializza lo StandardScaler
    scaler = StandardScaler()
    
    # Standardizza le features di addestramento
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Crea il modello di regressione lineare
    model = LinearRegression()
    
    # Addestra il modello
    model.fit(X_train_scaled, y_train)
    
    return model, scaler


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    return mae, mse, rmse, r2

def cross_validate_model(model, X, y, cv=10):
    k_fold = KFold(cv)
    return cross_val_score(model, X, y, cv=k_fold, n_jobs=1)

def visualize_linear_regression_results(model, X_test, y_test):
    DataVisualization.plot_linear_regression_results(X_test, y_test, model.predict(X_test))

def visualize_cross_validation_results(model, X, y):
    DataVisualization.plot_cross_validation_results(X, y, model)
