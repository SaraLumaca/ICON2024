import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

#Migliora le prestazioni del random forest e ottiene le previsioni

# Caricare il dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv')
data = pd.read_csv(file_path)

# Filtrare i dati per l'Italia
data_italy = data[data['Entity'] == 'Italy']

# Elenco delle patologie e dei rispettivi DALYs
dalys_columns = [
    'DALYs Cause: Depressive disorders', 'DALYs Cause: Schizophrenia',
    'DALYs Cause: Bipolar disorder', 'DALYs Cause: Eating disorders',
    'DALYs Cause: Anxiety disorders'
]

# Elenco di tutte le metriche presenti nel dataset
all_columns = data.columns.tolist()
# Rimuovere le colonne che non sono metriche (Entity, Code, Year)
feature_columns = [col for col in all_columns if col not in ['Entity', 'Code', 'Year'] + dalys_columns]

# Aggiungere l'anno come feature
feature_columns.append('Year')

# Funzione per eseguire la cross-validazione
def cross_validate_model(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    mse_scores = -cv_results
    rmse_scores = np.sqrt(mse_scores)
    return rmse_scores.mean(), rmse_scores.std()

# Funzione per prevedere i DALYs di una specifica patologia utilizzando tutte le metriche come feature
def predict_dalys(target_daly, feature_columns, data):
    # Selezionare le feature (tutte le metriche) e il target (DALYs)
    X = data[feature_columns].values
    y = data[target_daly].values

    # Suddivisione dei dati in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizzazione delle variabili
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definire il modello Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Eseguire la cross-validazione
    cv_rmse_mean, cv_rmse_std = cross_validate_model(rf, X_train_scaled, y_train)

    # Previsioni sui dati di test
    y_pred = rf.predict(X_test_scaled)

    # Calcolare le metriche di valutazione
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Valori di valutazione per {target_daly}:")
    print(f"R²: {r2}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"Cross-Validation RMSE Mean: {cv_rmse_mean}")
    print(f"Cross-Validation RMSE Std: {cv_rmse_std}")
    print()

    # Prevedere i DALYs futuri fino al 2030
    future_years = np.arange(data['Year'].max() + 1, 2031)
    last_known_year = data['Year'].max()
    last_known_data = data_italy[data_italy['Year'] == last_known_year].drop(columns=['Year', 'Entity', 'Code'])

    future_predictions = []
    current_features = last_known_data[feature_columns[:-1]].values

    for year in future_years:
        feature_row = np.append(current_features[-1], year).reshape(1, -1)
        feature_row_scaled = scaler.transform(feature_row)
        pred = rf.predict(feature_row_scaled)
        future_predictions.append(pred[0])

        # Aggiornare current_features con il nuovo valore previsto e aggiungere rumore
        new_features = current_features[-1].copy()
        new_features[-1] = pred[0]  # Aggiornare l'ultima feature con la previsione
        new_features += np.random.normal(0, 0.01, size=len(new_features))  # Aggiungere rumore a tutte le feature
        current_features = np.vstack([current_features, new_features])

    return future_years, np.array(future_predictions)

# Dizionario per memorizzare le previsioni future
future_predictions_dict = {}

# Prevedere i DALYs per ciascuna patologia
for target_daly in dalys_columns:
    future_years, future_predictions = predict_dalys(target_daly, feature_columns, data_italy)
    future_predictions_dict[target_daly] = future_predictions

    # Visualizzare i risultati
    plt.figure(figsize=(10, 6))
    plt.plot(data_italy['Year'], data_italy[target_daly], label='Storico')
    plt.plot(future_years, future_predictions, label='Previsione', linestyle='--')
    plt.title(f'Previsione dei DALYs per {target_daly.split(":")[1].strip()}')
    plt.xlabel('Anno')
    plt.ylabel('DALYs')
    plt.legend()
    plt.show()

# Crea un dataframe per i risultati futuri
future_df = pd.DataFrame(future_predictions_dict, index=future_years)
future_df.index.name = 'Year'

# Salva le previsioni ottenute in un file CSV
output_file_path = os.path.join(base_dir, '..', 'Risultati', 'PrevisioniDalysItalia.csv')
future_df.to_csv(output_file_path)

print(f"Previsioni salvate nel file: {output_file_path}")
