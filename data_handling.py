# data_handling.py

#Il modulo `data_handling.py` fornisce funzioni per gestire i dati, in particolare per la lettura di un file CSV e per generare una tabella di descrizione dei dati contenuti in un DataFrame pandas.
#Ecco cosa fanno le funzioni nel modulo:
#1. `read_data(file_path)`: Questa funzione accetta il percorso di un file CSV come argomento e utilizza la libreria pandas per leggere il file e restituire un DataFrame contenente i dati. Ad esempio, se chiami questa funzione con `read_data("data.csv")`, restituirà un DataFrame con i dati contenuti nel file "data.csv".
#2. `describe_data(df)`: Questa funzione accetta un DataFrame pandas come argomento e genera una tabella di descrizione dei dati. 
# La tabella contiene le seguenti informazioni per ciascuna variabile nel DataFrame: nome della variabile, tipo di dati, conteggio delle osservazioni non mancanti, numero di valori unici e numero di valori mancanti. 
# La tabella viene restituita come un DataFrame pandas.
#In sintesi, questo modulo fornisce funzioni di base per la gestione dei dati, consentendo di leggere dati da file CSV e di generare una tabella di descrizione per esaminare le caratteristiche dei dati.
#

import pandas as pd

def read_data(file_path):
    return pd.read_csv(file_path)

def describe_data(df):
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
