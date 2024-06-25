import os
import pandas as pd

# Function to describe dataframe
def describe(df):
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []

    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())

    output = pd.DataFrame({
        'variable': variables,
        'dtype': dtypes,
        'count': count,
        'unique': unique,
        'missing value': missing
    })

    return output

# Function to remove entities from dataframe
def remove_entities(df, entities_to_remove):
    return df[~df['Entity'].isin(entities_to_remove)]

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths for CSV files
file1 = os.path.join(current_dir, '..', 'DbOriginali', '1- mental-illnesses-prevalence.csv')
file2 = os.path.join(current_dir, '..', 'DbOriginali', '2- burden-disease-from-each-mental-illness(1).csv')
file4 = os.path.join(current_dir, '..', 'DbOriginali', '4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv')
file5 = os.path.join(current_dir, '..', 'DbOriginali', '5- anxiety-disorders-treatment-gap.csv')
file7 = os.path.join(current_dir, '..', 'DbOriginali', '7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv')
filesec = os.path.join(current_dir, '..', 'DbOriginali', 'SocialEconomicData.csv')

# Dictionary of file paths
file_paths = {
    'file1': file1,
    'file2': file2,
    'file4': file4,
    'file5': file5,
    'file7': file7,
    'filesec': filesec
}

# Check if all files exist
if all(os.path.exists(path) for path in file_paths.values()):

    # Read CSV files
    df1 = pd.read_csv(file_paths['file1'])
    df2 = pd.read_csv(file_paths['file2'])
    df4 = pd.read_csv(file_paths['file4'])
    df5 = pd.read_csv(file_paths['file5'])
    df7 = pd.read_csv(file_paths['file7'])
    df_sec = pd.read_csv(file_paths['filesec'])

    # Rename columns in df1 and df2
    columns_df1 = {
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }
    columns_df2 = {
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Depressive disorders': 'DALYs Cause: Depressive disorders',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Schizophrenia': 'DALYs Cause: Schizophrenia',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Bipolar disorder': 'DALYs Cause: Bipolar disorder',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Eating disorders': 'DALYs Cause: Eating disorders',
        'DALYs (rate) - Sex: Both - Age: Age-standardized - Cause: Anxiety disorders': 'DALYs Cause: Anxiety disorders'
    }

    df1.rename(columns=columns_df1, inplace=True)
    df2.rename(columns=columns_df2, inplace=True)

    # Merge df1 and df2
    df_merge = pd.merge(df1, df2, on=['Entity', 'Code', 'Year'], how='inner')

    # Entities to remove from merge
    entities_to_remove_merge = ['World', 'Africa (IHME GBD)', 'America (IHME GBD)', 'Asia (IHME GBD)', 'Europe (IHME GBD)', 'European Union (27)', 'Australia', 'Low-income countries', 'High-income countries', 'Lower-middle-income countries']
    df_merge = remove_entities(df_merge, entities_to_remove_merge)

    # Output file path for merged data
    output_file_path_merge = os.path.join(current_dir, '..', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv')
    df_merge.to_csv(output_file_path_merge, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_merge}")

    # Remove entities from df4 and df5
    entities_to_remove_4 = ['World']
    entities_to_remove_5 = ['Beijing/Shanghai, China', 'High-income countries', 'Lower-middle-income countries', 'Medellin, Colombia', 'Murcia, Spain', 'Sao Paulo, Brazil', 'Upper-middle-income countries']
    df4 = remove_entities(df4, entities_to_remove_4)
    df5 = remove_entities(df5, entities_to_remove_5)

    # Drop 'Code' column from df4 and df7
    df4.drop(columns=['Code'], inplace=True)
    df7.drop(columns=['Code'], inplace=True)

    # Output file paths for df4, df5, df7
    output_file_path_4 = os.path.join(current_dir, '..', 'DbDefinitivi', '4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv')
    output_file_path_5 = os.path.join(current_dir, '..', 'DbDefinitivi', '5- anxiety-disorders-treatment-gap.csv')
    output_file_path_7 = os.path.join(current_dir, '..', 'DbDefinitivi', '7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv')

    df4.to_csv(output_file_path_4, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_4}")
    df5.to_csv(output_file_path_5, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_5}")
    df7.to_csv(output_file_path_7, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_7}")

    # Rename columns in df_sec and save to CSV
    df_sec.rename(columns={
        'Schizophrenia disorders': 'Schizophrenia',
        'Depressive disorders': 'Depressive',
        'Anxiety disorders': 'Anxiety',
        'Bipolar disorders': 'Bipolar',
        'Eating disorders': 'Eating',
        'Country Name': 'Entity',  
        'Country Code': 'Code'
    }, inplace=True)

    output_file_path_dfe = os.path.join(current_dir, '..', 'DbDefinitivi', 'SocialEconomicData.csv')
    df_sec.to_csv(output_file_path_dfe, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_dfe}")

else:
    print(f"Uno dei file non esiste.")
