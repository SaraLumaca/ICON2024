import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from clustering_analysis_integration import ClusteringAnalysis
import os

# Determina il percorso del file corrente (main.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Costruisci i percorsi relativi utilizzando os.path.join
data_df1_path = os.path.join(current_dir, 'SocialEconomicData.csv')
data_df2_path = os.path.join(current_dir, '..', 'Europa', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv')
#dataset_arricchito_completo_path = os.path.join(current_dir, '..', 'Europa', 'Risultati', 'dataset_arricchito_completo.csv')

# Carica i dataset
data_df1 = pd.read_csv(data_df1_path)
data_df2 = pd.read_csv(data_df2_path)
#dataset_arricchito_completo = pd.read_csv(dataset_arricchito_completo_path)

# Verifica la presenza delle colonne
print("Colonne in data_df1:", data_df1.columns)
print("Colonne in data_df2:", data_df2.columns)
#print("Colonne in dataset_arricchito_completo:", dataset_arricchito_completo.columns)

# Unisci i dataset
data_df2.rename(columns={"Code": "Country Code"}, inplace=True)
merged_df = pd.merge(data_df1, data_df2, on="Country Code", how="inner")

# Pulisci i dati
cleaned_df = merged_df.drop_duplicates()

# Rinomina le colonne per chiarezza
clustering_df = cleaned_df.copy()
clustering_df.rename(columns={
    'Schizophrenia disorders': 'Schizophrenia',
    'Depressive disorders': 'Depressive',
    'Anxiety disorders': 'Anxiety',
    'Bipolar disorders': 'Bipolar',
    'Eating disorders': 'Eating',
}, inplace=True)

# Rinomina la colonna 'Country Name' a 'Entity' per corrispondere con dataset_arricchito_completo
#clustering_df.rename(columns={'Country Name': 'Entity'}, inplace=True)

# Stampa le colonne di clustering_df per verificare
print("Colonne in clustering_df dopo la pulizia:", clustering_df.columns)

# Scala i dati prima del clustering
scaler = StandardScaler()
clustering_df[['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']] = scaler.fit_transform(
    clustering_df[['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']])

# Ricalcola il GDP medio utilizzando gli anni dal 1990 al 2019
subset_gdp_columns = ['1990 [YR1990]', '2000 [YR2000]', '2014 [YR2014]', '2015 [YR2015]',
                      '2016 [YR2016]', '2017 [YR2017]', '2018 [YR2018]', '2019 [YR2019]']

for col in subset_gdp_columns:
    clustering_df[col] = pd.to_numeric(clustering_df[col].replace('..', pd.NA), errors='coerce')

clustering_df['Average_GDP'] = clustering_df[subset_gdp_columns].mean(axis=1, skipna=True)

# Istanziamento della classe di analisi del clustering
clustering_analysis = ClusteringAnalysis()

# Esegui il clustering KMeans sui dati completi
kmeans_clustered_df = clustering_analysis.kmeans_clustering(clustering_df)

# Aggiungi i risultati del clustering al dataset completo
clustering_df['Cluster_KMeans'] = kmeans_clustered_df['Cluster_KMeans']



# Descrizioni
cluster_descriptions = {
    0: "0",
    1: "1",
    2: "2"
}

# Aggiunge una colonna per la descrizione
clustering_df['Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")'] = clustering_df['Cluster_KMeans'].map(cluster_descriptions)

# Rinomina la colonna 'Country Name' a 'Entity' per corrispondere con dataset_arricchito_completo
clustering_df.rename(columns={'Country Name': 'Entity'}, inplace=True)

clustering_df = clustering_df.loc[:, ~clustering_df.columns.duplicated()]

print("Colonne in clustering_df:", clustering_df.columns)

# Unisci il dataset esistente con i nuovi dati di clustering
dataset_finale = pd.merge(data_df2, clustering_df[['Entity', 'Cluster_KMeans', 'Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")']], on='Entity', how='left')

# Salva il dataset arricchito
dataset_finale.to_csv(data_df2_path, index=False)

print("Dataset aggiornato e salvato con successo.")
