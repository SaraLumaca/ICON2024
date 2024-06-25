import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from clustering_analysis_integration import ClusteringAnalysis
import os

# Percorsi relativi utilizzando os.path.join

current_dir = os.path.dirname(os.path.abspath(__file__))

data_df1_path = os.path.join(current_dir, 'SocialEconomicData.csv')
data_df2_path = os.path.join(current_dir, '..', 'Europa', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo.csv')

# Carica i dataset
SED_df1 = pd.read_csv(data_df1_path)
Dalys_df2 = pd.read_csv(data_df2_path)

#stampa
print("Colonne in SED_df1:", SED_df1.columns)
print("Colonne in Dalys_df2:", Dalys_df2.columns)

# Unisci i dataset
Dalys_df2.rename(columns={"Code": "Country Code"}, inplace=True)
merged_df = pd.merge(SED_df1, Dalys_df2, on="Country Code", how="inner")

print("Colonne in merged_df:", merged_df.columns)

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

# Istanzia la classe di analisi del clustering
clustering_analysis = ClusteringAnalysis()

# clustering KMeans sui dati completi
kmeans_clustered_df = clustering_analysis.kmeans_clustering(clustering_df)

# Aggiungi i risultati del clustering al dataset completo
clustering_df['Cluster_KMeans'] = kmeans_clustered_df['Cluster_KMeans']

# Numero del cluster
cluster = {
    0: "0",
    1: "1",
    2: "2"
}

# Aggiunge una colonna per la descrizione
clustering_df['Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")'] = clustering_df['Cluster_KMeans'].map(cluster)

# Rinomina la colonna 'Country Name' a 'Entity' per corrispondere l'altro dataset
clustering_df.rename(columns={'Country Name': 'Entity'}, inplace=True)

#elimina i duplicati
clustering_df = clustering_df.loc[:, ~clustering_df.columns.duplicated()]

# Unisci il dataset esistente con i nuovi dati di clustering
dataset_finale = pd.merge(Dalys_df2, clustering_df[['Entity', 'Year', 'Cluster_KMeans', 'Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")']],
                          on=['Entity', 'Year'], how='left')

dataset_finale_path = os.path.join(current_dir, '..', 'Europa', 'DbDefinitivi', 'DisturbiMentali-DalysNazioniDelMondo-GruppoDiIntervento.csv')
dataset_finale.to_csv(dataset_finale_path, index=False)

print("Dataset aggiornato e salvato con successo.")
