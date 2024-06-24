#MAIN
#if choice == "111":
#                    diseases = [
#                        'Schizophrenia disorders',
#                        'Depressive disorders',
#                        'Anxiety disorders',
#                        'Bipolar disorders',
 #                       'Eating disorders'
#                    ]
#                    # Eseguire la regressione lineare per ogni patologia
 #                   for disease in diseases:
 #                       lr.linear_regression_analysis(df1,disease)
 #                       lr.print_histogram_interpretation()




import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing, metrics
import numpy as np
from sklearn.linear_model import LassoCV

def plot_prediction_errors(y_test, y_preds, dependent_variable):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey=True)
    fig.suptitle(f'Comparazione della distribuzione degli errori per {dependent_variable}', fontsize=20)

    model_names = list(y_preds.keys())
    for ax, model_name in zip(axes.flatten(), model_names):
        errors = y_test - y_preds[model_name]
        ax.hist(errors, bins=30, alpha=0.7, color='blue')
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    
def linear_regression_analysis(df1, dependent_variable):
    # Selezione delle colonne necessarie per l'analisi
    features = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders',
                'Eating disorders']
    features.remove(dependent_variable)

    X_model = df1[features]
    y_model = df1[dependent_variable]

    # Normalizzazione delle features
    scaler = preprocessing.MinMaxScaler()
    X_model_norm = scaler.fit_transform(X_model)

    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_model_norm, y_model, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "LassoCV Regression": LassoCV(cv=10)  # Aggiunta di LassoCV
    }

    best_model_name = None
    best_score = -np.inf
    results = []
    y_preds = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_preds[model_name] = y_pred
        
        # Metriche di valutazione
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(y_test, y_pred)

        # Validazione incrociata a 10 fold
        k_fold = KFold(10)
        cv_scores = cross_val_score(model, X_model_norm, y_model.to_numpy(), cv=k_fold, n_jobs=1)
        mean_cv_score = np.mean(cv_scores)

        # Salvataggio dei risultati
        results.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "CV Score": mean_cv_score,
            "Coefficients": model.coef_,
            "Intercept": model.intercept_
        })

        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_model_name = model_name

        # Stampa dei risultati
        print(f"\n{model_name} Results:")
        print("Coefficients:")
        for feature, coef in zip(features, model.coef_):
            print(f"{feature}: {coef:.4f}")
        print(f"Intercept: {model.intercept_:.4f}\n")
        print("Mean Absolute Error (MAE): ", mae)
        print("Mean Squared Error (MSE): ", mse)
        print("Root Mean Squared Error (RMSE): ", rmse)
        print("R2 Score: ", r2)
        print("Cross-validation scores: ", cv_scores)
        print("Mean Cross-validation score: ", mean_cv_score)

    print(f"\nIl modello migliore per questo set è: {best_model_name} con un punteggio di cross-validazione medio di {best_score:.4f}")

    # Visualizzazione degli errori per ogni modello
    for result in results:
        model_name = result["Model"]
        y_pred = models[model_name].predict(X_test)
        plot_prediction_errors(y_test, y_preds, dependent_variable)
        


def print_histogram_interpretation():
    print(""" 
----------------DESCRIZIONE DEL GRAFICO ------------------------------

Per ogni punto dati nel set di test, si calcola l'errore di previsione.
Questi errori sono poi raccolti per formare una distribuzione.

Costruzione dell'Istogramma:

- Gli errori di previsione calcolati sono suddivisi in intervalli (bins).
- Si conta il numero di errori che rientrano in ciascun intervallo e si tracciano questi conteggi sull'asse delle Y dell'istogramma.
- L'asse delle X rappresenta l'intervallo di valori degli errori di previsione.

Interpretazione dell'Istogramma:

- Centro della Distribuzione: Se la maggior parte degli errori è concentrata intorno a zero, significa che il modello è accurato e non ha un bias significativo.
- Distribuzione degli Errori: La forma della distribuzione può indicare la presenza di errori sistematici. Una distribuzione simmetrica e stretta intorno a zero indica un buon modello.
- Se ci sono molti errori grandi (sia positivi che negativi), indica che il modello potrebbe avere difficoltà con alcuni dati particolari.
    """)
