import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(*dfs):
    """
    Visualizza le matrici di correlazione per i dataframe forniti.

    Args:
        *dfs: Argomenti variabili contenenti i dataframe Pandas.

    Returns:
        None
    """
    for df in dfs:
        # Identifica le colonne con valori non numerici
        non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
        df_numeric = df.drop(non_numeric_columns, axis=1)

        # Gestisci i valori non numerici
        df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')

        # Calcola la matrice di correlazione
        corr_matrix = df_numeric.corr()
        
        # Visualizza la matrice di correlazione utilizzando Seaborn
        plt.figure(figsize=(10, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title(f'Matrice di correlazione per {df.name}')
        plt.show()

       