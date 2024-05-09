import data_handling
import data_visualization
import data_analysis
import correlation_matrix
from sklearn.model_selection import train_test_split
import neural_network_model
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd


def main():
    # Ottieni il percorso del file CSV utilizzando un percorso relativo
    current_dir = os.path.dirname(__file__)  # Ottieni la directory corrente (dove si trova main.py)
    file_path1 = os.path.join(current_dir, '..', 'dbUsati', 'DisturbiMentali-DalysNazioniDelMondo.csv')
    file_path2 = os.path.join(current_dir, '..', 'dbUsati', 'DisturbiMentali2008-areeGeografiche.csv')
    file_path3 = os.path.join(current_dir, '..', 'dbUsati', '6- depressive-symptoms-across-us-population.csv')
    file_path4 = os.path.join(current_dir, '..', 'dbUsati', '7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv')

    # Leggi i dati dai file CSV
    df1 = data_handling.read_data(file_path1)
    df2 = data_handling.read_data(file_path2)
    df3 = data_handling.read_data(file_path3)
    df4 = data_handling.read_data(file_path4)
    # Menu
    while True:
        print("Scegli un'opzione:")
        print("1. Descrizione dei dati")
        print("2. Visualizza le matrici di correlazione")
        print("3. Regressione lineare")
        print("4. Reti neurali")
        print("5. Esci")
        choice = input("Scelta: ")

        if choice == '1':
            # Effettua la descrizione dei dati
            print("Descrizione tabella1: Disturbi Mentali-DALYs in tutte le nazioni del mondo (Anni:1990-2019)")
            print(data_handling.describe_data(df1))
            print("\n")
            print("Descrizione tabella2: Disturbi mentali in diverse aree geografiche(anno:2008)")
            print(data_handling.describe_data(df2))
            print("\n")
            print("Descrizione tabella3: Sintomi depressivi")
            print(data_handling.describe_data(df3))
            print("\n")
            print("Descrizione tabella4: Numero di nazioni con dati primari ")
            print(data_handling.describe_data(df4))

        elif choice == '2':
            correlation_matrix.correlation_matrix(df1, df2, df3, df4)  #Quando chiudi una finestra si apre quella dopo

        elif choice == '3':
            # Prepara i dati per l'analisi utilizzando data_analysis
            X_model_norm, y_model = data_analysis.prepare_data_for_modeling(df1, ['Schizophrenia disorders',
                                                                                  'Depressive disorders',
                                                                                  'Anxiety disorders',
                                                                                  'Bipolar disorders'],
                                                                            'Eating disorders')

            # Addestra un modello di regressione lineare
            linear_model, scaler = data_analysis.train_linear_regression_model(X_model_norm, y_model)

            # Valuta il modello
            mae, mse, rmse, r2 = data_analysis.evaluate_model(linear_model, X_model_norm, y_model)

            print("\n")
            print("Errore Assoluto Medio del Modello Lineare: ", mae)
            print("Errore Quadratico Medio del Modello Lineare: ", mse)
            print("Radice dell'Errore Quadratico Medio del Modello Lineare: ", rmse)
            print("Punteggio R2 del Modello Lineare: ", r2)

            # Visualizza il grafico della regressione lineare
            data_visualization.plot_scatter_chart(y_model, linear_model.predict(X_model_norm), "Regressione Lineare",
                                                  "Target", "Predizione")

            # Esegui la validazione incrociata del modello
            cv_scores = data_analysis.cross_validate_model(linear_model, X_model_norm, y_model)
            print("Cross validation del modello lineare ", cv_scores)

        elif choice == '4':
            # Utilizza il modello di rete neurale
            features = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders']
            target = 'Eating disorders'

            # Prepara i dati per l'addestramento utilizzando data_analysis
            X, y = data_analysis.prepare_data_for_modeling(df1, features, target)

            # Dividere i dati in set di training e test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Creare e addestrare il modello
            neural_model = neural_network_model.train(X_train, y_train)

            # Valutare il modello
            mae_nn, mse_nn, rmse_nn, r2_nn = neural_network_model.evaluate(neural_model, X_test, y_test)
            print("\n")
            print("Errore Assoluto Medio del Modello di Rete Neurale è: ", mae_nn)
            print("Errore Quadratico Medio del Modello di Rete Neurale è: ", mse_nn)
            print("Radice dell'Errore Quadratico Medio del Modello di Rete Neurale è: ", rmse_nn)
            print("Punteggio R2 del Modello di Rete Neurale è: ", r2_nn)

            # Visualizzare i risultati del modello
            neural_network_model.visualize_results(neural_model, X_test, y_test)

        elif choice == '5':
            print("Arrivederci!")
            break
        else:
            print("Scelta non valida. Si prega di scegliere un'opzione valida.")


if __name__ == "__main__":
    main()
