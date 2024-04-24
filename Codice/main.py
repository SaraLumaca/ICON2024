import data_handling
import data_visualization
import data_analysis
from sklearn.model_selection import train_test_split
import neural_network_model

def main():
    # Leggi i dati dai file CSV
    df1 = data_handling.read_data(r'C:\Users\letha\Desktop\Salute mentale nel mondo - Copia\dbOriginale copy\DisturbiMentali-DalysNazioniDelMondo.csv')
    df2 = data_handling.read_data(r'C:\Users\letha\Desktop\Salute mentale nel mondo - Copia\dbOriginale copy\DisturbiMentali2008-areeGeografiche.csv')
    df3 = data_handling.read_data(r'C:\Users\letha\Desktop\Salute mentale nel mondo\dbOriginale\6- depressive-symptoms-across-us-population.csv')
    df4 = data_handling.read_data(r'C:\Users\letha\Desktop\Salute mentale nel mondo\dbOriginale\7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv')
    
    # Effettua la descrizione dei dati
    print("Descrizione tabella1: Disturbi Mentali-DALYs in tutte le nazioni del mondo (Anni:1990-2019)")
    print(data_handling.describe_data(df1))
    print("\n")
    print("Descrizione tabella2: Disturbi mentali in diverse aree geografiche(anno:2008)")
    print(data_handling.describe_data(df2))
    print("\n")
    print("Descrizione tabella3:Sintomi depressivi")
    print(data_handling.describe_data(df3))
    print("\n")
    print("Descrizione tabella4: Numero di nazioni con dati primari ")
    print(data_handling.describe_data(df4))
    
    # Visualizza grafici con Plotly
    #data_visualization.plot_bar_chart(df2, "Major depression", "Entity", color_column="Bipolar disorder")
    #data_visualization.plot_bar_chart(df2, "Major depression", "Entity", color_column="Eating disorders")
    
    #data_visualization.plot_bar_line_chart(df2, "Entity", "Major depression", "Bipolar disorder")
    #data_visualization.plot_bar_line_chart(df2, "Entity", "Major depression", "Eating disorders")
    #data_visualization.plot_bar_line_chart(df2, "Entity", "Major depression", "Anxiety disorders")

    
    # Prepara i dati per l'analisi utilizzando data_analysis
    X_model_norm, y_model = data_analysis.prepare_data_for_modeling(df1, ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders'], 'Eating disorders')
    
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
    data_visualization.plot_scatter_chart(y_model, linear_model.predict(X_model_norm), "Regressione Lineare", "Target", "Predizione")
    
    # Esegui la validazione incrociata del modello
    cv_scores = data_analysis.cross_validate_model(linear_model, X_model_norm, y_model)
    print("Cross validation del modello lineare ", cv_scores)
    
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

if __name__ == "__main__":
    main()
