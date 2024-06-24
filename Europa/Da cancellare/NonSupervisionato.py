import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import plotly.express as px

# Caricamento del dataset
def load_data():
    df = pd.read_csv("DisturbiMentali-DalysNazioniDelMondo.csv")
    df = df.rename(columns={'Schizophrenia disorders' : 'Schizophrenia', 
                            'Depressive disorders': 'Depressive',
                            'Anxiety disorders':'Anxiety',
                            'Bipolar disorders':'Bipolar',
                            'Eating disorders':'Eating'})
    return df

# Funzione per descrivere il dataframe
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

# Funzione per creare una matrice scatter
def scatter_matrix(df):
    fig = px.scatter_matrix(df, dimensions=["Schizophrenia", "Depressive", "Anxiety", 
                                            "Bipolar"], color="Eating")
    fig.show()

# Funzione per visualizzare la matrice di correlazione
def correlation_matrix(df):
    Numerical = ['Schizophrenia', 'Depressive','Anxiety','Bipolar','Eating']
    Corrmat = df[Numerical].corr()
    plt.figure(figsize=(10, 5), dpi=200)
    sns.heatmap(Corrmat, annot=True, fmt=".2f", linewidth=.5)
    plt.show()

# Funzione per eseguire il clustering agglomerativo
def agglomerative_clustering(df):
    features = ['Schizophrenia', 'Depressive','Anxiety','Bipolar','Eating']
    cluster_agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    cluster_agg.fit(df[features])
    labels = cluster_agg.labels_

    plot_clusters(df, features, labels, 'Agglomerative Clustering')
    
    # Stampa delle nazioni per ciascun cluster
    # Stampa delle nazioni per ciascun cluster
    df['Cluster'] = labels
    for cluster in range(3):
        print(f"Nazioni nel cluster {cluster}:")
        print(df[df['Cluster'] == cluster]['Entity'].unique())
        print("\n")
    
    
    plot_clusters(df, features, labels, 'Agglomerative Clustering')

# Funzione per eseguire il clustering KMeans
def kmeans_clustering(df):
    features = ['Schizophrenia', 'Depressive','Anxiety','Bipolar','Eating']
    kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)

    # Calcolo dell'inerzia e dei centri di cluster
    iner = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
        kmeans.fit(df[features])
        iner.append(kmeans.inertia_)

    # Plot del grafico dell'inerzia
    plt.figure(figsize=(10,5), dpi=200)
    plt.plot(range(1, 10), iner, color='purple')
    plt.xticks(range(1, 10))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.axvline(x=3, color='b', label='axvline - full height', linestyle="dashed")
    plt.show()

    # Utilizzo di KneeLocator per trovare il gomito
    KL = KneeLocator(range(1,10), iner, curve="convex", direction="decreasing")
    print("Optimal number of clusters (elbow method):", KL.elbow)

    # Calcolo del coefficiente di silhouette per diversi numeri di cluster
    silhouette_coefficients = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, init="random", n_init=10, max_iter=300, random_state=42)
        kmeans.fit(df[features])
        score = silhouette_score(df[features], kmeans.labels_)
        silhouette_coefficients.append(score)

    # Plot del coefficiente di silhouette
    plt.figure(figsize=(10,5), dpi=200)
    plt.plot(range(2, 11), silhouette_coefficients, color='purple')
    plt.xticks(range(2, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()

    # Esecuzione del clustering KMeans con 3 cluster
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df[features])
    labels_Km = kmeans.labels_

    df['Cluster'] = labels_Km
    for cluster in range(3):
        print(f"Nazioni nel cluster {cluster}:")
        print(df[df['Cluster'] == cluster]['Entity'].unique())
        print("\n")
    plot_clusters(df, features, labels_Km, 'KMeans Clustering')

# Funzione per creare scatter plot con clustering
def plot_clusters(df, features, labels, title_prefix):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    sns.scatterplot(ax=axes[0], data=df, x='Schizophrenia', y='Depressive').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Schizophrenia', y='Depressive', hue=labels).set_title(f'{title_prefix} - Schizophrenia vs Depressive')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    sns.scatterplot(ax=axes[0], data=df, x='Depressive', y='Anxiety').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Depressive', y='Anxiety', hue=labels).set_title(f'{title_prefix} - Depressive vs Anxiety')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    sns.scatterplot(ax=axes[0], data=df, x='Anxiety', y='Bipolar').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Anxiety', y='Bipolar', hue=labels).set_title(f'{title_prefix} - Anxiety vs Bipolar')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    sns.scatterplot(ax=axes[0], data=df, x='Bipolar', y='Eating').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Bipolar', y='Eating', hue=labels).set_title(f'{title_prefix} - Bipolar vs Eating')
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    sns.scatterplot(ax=axes[0], data=df, x='Eating', y='Schizophrenia').set_title('Without clustering')
    sns.scatterplot(ax=axes[1], data=df, x='Eating', y='Schizophrenia', hue=labels).set_title(f'{title_prefix} - Eating vs Schizophrenia')

    plt.show()

# Menù a riga di comando
def menu():
    df = load_data()
    while True:
        print("\nMenù:")
        print("1. Descrivere il dataframe")
        print("2. Creare una matrice scatter")
        print("3. Visualizzare la matrice di correlazione")
        print("4. Eseguire clustering agglomerativo")
        print("5. Eseguire clustering KMeans")
        print("6. Esci")
        
        choice = input("Scegli un'opzione (1-6): ")
        
        if choice == '1':
            print(describe(df))
        elif choice == '2':
            scatter_matrix(df)
        elif choice == '3':
            correlation_matrix(df)
        elif choice == '4':
            agglomerative_clustering(df)
        elif choice == '5':
            kmeans_clustering(df)
        elif choice == '6':
            print("Uscita...")
            break
        else:
            print("Scelta non valida, riprova.")

if __name__ == "__main__":
    menu()
