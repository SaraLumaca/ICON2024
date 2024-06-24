import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx
import matplotlib.pyplot as plt


# Funzione per caricare il dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Funzione per esplorare le colonne con sinonimi e relazioni
def explore_ontology_columns(dataset):
    synonym_columns = [col for col in dataset.columns if 'Sinonimi' in col or 'Relazioni' in col]
    return synonym_columns

# Funzione per creare nuovi indicatori derivati dalle relazioni ontologiche
def create_derived_indicators(dataset):
    dataset['Disturbi Cognitivi'] = dataset['Anxiety disorders']
    dataset['Disturbi dell\'Umore'] = dataset['Depressive disorders'] + dataset['Bipolar disorders']
    return dataset

# Funzione per calcolare la matrice di correlazione
def calculate_correlation_matrix(dataset):
    correlation_matrix = dataset[['Schizophrenia disorders', 'Depressive disorders', 
                                  'Anxiety disorders', 'Bipolar disorders', 
                                  'Eating disorders', 'Disturbi Cognitivi', 
                                  'Disturbi dell\'Umore']].corr()
    return correlation_matrix

# Funzione per visualizzare la matrice di correlazione
def plot_correlation_matrix(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrice di Correlazione tra i Disturbi')
    plt.show()

# Funzione per visualizzare le tendenze temporali dei disturbi in Italia
def plot_trends_italy(dataset):
    italy_data = dataset[dataset['Entity'] == 'Italy']
    plt.figure(figsize=(12, 6))
    plt.plot(italy_data['Year'], italy_data['Disturbi Cognitivi'], label='Disturbi Cognitivi')
    plt.plot(italy_data['Year'], italy_data['Disturbi dell\'Umore'], label='Disturbi dell\'Umore')
    plt.xlabel('Anno')
    plt.ylabel('Prevalenza')
    plt.title('Andamento dei Disturbi Cognitivi e dell\'Umore nel Tempo (Italia)')
    plt.legend()
    plt.show()
    


def create_ontology_network(dataset):
    G = nx.Graph()
    
    # Aggiungere nodi per ogni disturbo
    disorders = ['Schizophrenia', 'Depressive', 'Anxiety', 'Bipolar', 'Eating']
    for disorder in disorders:
        G.add_node(disorder)
    
    # Aggiungere relazioni ontologiche
    relations = {
        'Anxiety': ['Disturbi Cognitivi'],
        'Depressive': ['Disturbi dell\'Umore'],
        'Bipolar': ['Disturbi dell\'Umore'],
        # Aggiungere altre relazioni se disponibili
    }
    
    for disorder, related_disorders in relations.items():
        for related_disorder in related_disorders:
            G.add_edge(disorder, related_disorder)
    
    # Visualizzare il grafico a rete
    pos = nx.spring_layout(G,k=0.5)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=10, font_weight='bold')
    plt.title('Relazioni Ontologiche tra Disturbi Mentali')
    plt.show()


# Funzione principale per eseguire tutte le analisi
def main():
    # Caricare il dataset
    file_path="dataset_arricchito_completo.csv"
    dataset = load_dataset(file_path)
    
    # Esplorare le colonne con sinonimi e relazioni
    synonym_columns = explore_ontology_columns(dataset)
    print("Colonne con sinonimi e relazioni:", synonym_columns)
    
    # Creare nuovi indicatori derivati
    dataset = create_derived_indicators(dataset)
    
    # Calcolare la matrice di correlazione
    correlation_matrix = calculate_correlation_matrix(dataset)
    
    # Visualizzare la matrice di correlazione
    plot_correlation_matrix(correlation_matrix)
    
    # Visualizzare le tendenze temporali dei disturbi in Italia
    plot_trends_italy(dataset)
    
    # Esegui la creazione del grafico a rete
    create_ontology_network(dataset)

# Eseguire la funzione principale
main()
