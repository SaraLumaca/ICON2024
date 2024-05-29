import os
import pandas as pd

# file 4
file_path4 = r'C:\Users\marti\Documents\GitHub\ICON2024\DbOriginali\4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'

if os.path.exists(file_path4):

    df4 = pd.read_csv(file_path4)

    columns_df4 = {
        'Entity'
        'Code'
        'Year'
        'Major depression'
        'Bipolar disorder'
        'Eating disorders'
        'Dysthymia'
        'Schizophrenia'
        'Anxiety disorders'
    }

    entities_to_remove = ['World']

    df4 = df4[~df4['Entity'].isin(entities_to_remove)]

    df4.drop(columns=['Code'], inplace=True)

    output_file_path_4 = r'C:\Users\marti\Documents\GitHub\ICON2024\DbDefinitivi\4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv'

    df4.to_csv(output_file_path_4, index=False)

    print(f"Il file è stato salvato con successo in: {output_file_path_4}")
else:
    print(f"Il file {file_path4} non esiste.")