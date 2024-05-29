import os
import pandas as pd

# Percorso del primo file CSV
file_path1 = r'C:\Users\marti\Documents\GitHub\ICON2024\DbOriginali\1- mental-illnesses-prevalence.csv'

# Percorso del secondo file CSV
file_path2 = r'C:\Users\marti\Documents\GitHub\ICON2024\DbOriginali\2- burden-disease-from-each-mental-illness(1).csv'

# Verifica se entrambi i file esistono
if os.path.exists(file_path1) and os.path.exists(file_path2):
    # Leggi i file CSV
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Identifica la chiave di unione (es. 'Entity', 'Code', 'Year')
    chiavi_di_unione = ['Entity', 'Code', 'Year']

    # Rinomina le colonne nei DataFrame
    columns_df1 = {
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }

    df1.rename(columns=columns_df1, inplace=True)

    columns_df2 = {
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'DALYs Cause: Depressive disorders',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia': 'DALYs Cause: Schizophrenia',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder': 'DALYs Cause: Bipolar disorder',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders': 'DALYs Cause: Eating disorders',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders': 'DALYs Cause: Anxiety disorders'
    }

    df2.rename(columns=columns_df2, inplace=True)

    # Esegui il merge dei DataFrame
    df_merge = pd.merge(df1, df2, on=chiavi_di_unione, how='inner')

    # Elimina le righe con le entità specificate
    entities_to_remove = ['World', 'Africa (IHME GBD)', 'America (IHME GBD)', 'Asia (IHME GBD)', 'Europe (IHME GBD)', 'European Union (27)', 'Australia', 'Low-income countries', 'High-income countries', 'Lower-middle-income countries']
    df_merge = df_merge[~df_merge['Entity'].isin(entities_to_remove)]

    # Percorso del file di output
    output_file_path_1 = r'C:\Users\marti\Documents\GitHub\ICON2024\DbDefinitivi\DisturbiMentali-DalysNazioniDelMondo.csv'

    # Salva il DataFrame risultante nel file di output
    df_merge.to_csv(output_file_path_1, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_1}")
else:
    print(f"Il file {file_path1} o {file_path2} non esiste.")

