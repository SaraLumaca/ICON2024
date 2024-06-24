import pandas as pd
import requests

# Caricare il dataset
file_path = 'ICON-20240618-0005\Europa\Risultati\dataset_arricchito.csv'
dataset = pd.read_csv(file_path)

# Definire una funzione per simulare l'estrazione delle informazioni dalle pagine web
def extract_metadata(uri):
    if uri == "http://purl.obolibrary.org/obo/DOID_2030":
        return {
            "ID": "DOID:2030",
            "Nome": "disturbo d'ansia",
            "Definizione": "Un disturbo cognitivo che comporta una paura eccessiva e irrazionale delle situazioni quotidiane.",
            "Sinonimi": "ansia [ESATTO], stato d'ansia [ESATTO]",
            "Relazioni": "è_un disturbo cognitivo"
        }
    elif uri == "http://purl.obolibrary.org/obo/DOID_1596":
        return {
            "ID": "DOID:1596",
            "Nome": "disturbo depressivo",
            "Definizione": "Un disturbo mentale caratterizzato da umore depresso persistente o perdita di interesse per le attività.",
            "Sinonimi": "depressione [ESATTO], disturbo depressivo maggiore [ESATTO]",
            "Relazioni": "è_un disturbo mentale"
        }
    elif uri == "http://purl.obolibrary.org/obo/DOID_5419":
        return {
            "ID": "DOID:5419",
            "Nome": "schizofrenia",
            "Definizione": "Un disturbo mentale grave caratterizzato da distorsioni del pensiero, delle percezioni, delle emozioni e del comportamento.",
            "Sinonimi": "schizofrenia [ESATTO]",
            "Relazioni": "è_un disturbo mentale"
        }
    elif uri == "http://purl.obolibrary.org/obo/DOID_3312":
        return {
            "ID": "DOID:3312",
            "Nome": "disturbo bipolare",
            "Definizione": "Un disturbo mentale caratterizzato da episodi alternanti di umore elevato (mania) e umore depresso (depressione).",
            "Sinonimi": "bipolarismo [ESATTO], disturbo maniaco-depressivo [ESATTO]",
            "Relazioni": "è_un disturbo dell'umore"
        }
    elif uri == "http://purl.obolibrary.org/obo/DOID_8670":
        return {
            "ID": "DOID:8670",
            "Nome": "disturbo alimentare",
            "Definizione": "Un disturbo caratterizzato da comportamenti alimentari anormali che influenzano negativamente la salute fisica e mentale.",
            "Sinonimi": "disturbo del comportamento alimentare [ESATTO]",
            "Relazioni": "è_un disturbo del comportamento"
        }
    else:
        return {
            "ID": None,
            "Nome": None,
            "Definizione": None,
            "Sinonimi": None,
            "Relazioni": None
        }

# Applicare la funzione per estrarre le informazioni per ciascun URI
dataset['Anxiety Metadata'] = dataset['Anxiety disorders_URI'].apply(extract_metadata)
dataset['Depression Metadata'] = dataset['Depressive disorders_URI'].apply(extract_metadata)
dataset['Schizophrenia Metadata'] = dataset['Schizophrenia disorders_URI'].apply(extract_metadata)
dataset['Bipolar Metadata'] = dataset['Bipolar disorders_URI'].apply(extract_metadata)
dataset['Eating Metadata'] = dataset['Eating disorders_URI'].apply(extract_metadata)

# Espandere le informazioni estratte in colonne separate
metadata_cols = ['ID', 'Nome', 'Definizione', 'Sinonimi', 'Relazioni']
for col in metadata_cols:
    dataset[f'Anxiety {col}'] = dataset['Anxiety Metadata'].apply(lambda x: x[col])
    dataset[f'Depression {col}'] = dataset['Depression Metadata'].apply(lambda x: x[col])
    dataset[f'Schizophrenia {col}'] = dataset['Schizophrenia Metadata'].apply(lambda x: x[col])
    dataset[f'Bipolar {col}'] = dataset['Bipolar Metadata'].apply(lambda x: x[col])
    dataset[f'Eating {col}'] = dataset['Eating Metadata'].apply(lambda x: x[col])

# Rimuovere le colonne temporanee
dataset = dataset.drop(columns=['Anxiety Metadata', 'Depression Metadata', 'Schizophrenia Metadata', 'Bipolar Metadata', 'Eating Metadata'])

# Salvare il dataset arricchito
enriched_file_path = r'ICON-20240618-0005\Europa\Risultati\dataset_arricchito_completo.csv'
dataset.to_csv(enriched_file_path, index=False)

print("Dataset arricchito salvato come:", enriched_file_path)
