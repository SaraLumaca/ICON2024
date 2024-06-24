import pandas as pd
from rdflib import Graph

# Carica l'ontologia
ontology_path = r'ICON-20240618-0005\Europa\Ontologia\HumanDiseaseOntology.owl'
g = Graph()
g.parse(ontology_path, format='xml')

# Carica il dataset
dataset_path = r'ICON-20240618-0005\Europa\DbDefinitivi\DisturbiMentali-DalysNazioniDelMondo.csv'
df = pd.read_csv(dataset_path)

# Visualizza le colonne del DataFrame
print(df.columns)

def get_disorder_uri(disorder_label, parent_class):
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX obo: <http://purl.obolibrary.org/obo/>

    SELECT ?disorder
    WHERE {{
        ?disorder rdfs:label "{disorder_label}"@en .
        ?disorder rdfs:subClassOf* <{parent_class}> .
    }}
    """
    print(f"Debug: Query for '{disorder_label}' under '{parent_class}':")
    print(query)

    try:
        results = g.query(query)
        for row in results:
            return str(row['disorder'])
        print(f"No URI found for '{disorder_label}'")
        return 'Unknown'
    except Exception as e:
        print(f"Error in SPARQL query for '{disorder_label}': {e}")
        return 'Unknown'

# Definizione delle relazioni tra disturbi mentali e classi superiori nell'ontologia
ontology_disorders = {
    'Schizophrenia disorders': ('schizophrenia', 'http://purl.obolibrary.org/obo/DOID_2468'),  # psychotic disorder
    'Depressive disorders': ('depressive disorder', 'http://purl.obolibrary.org/obo/DOID_3324'),  # mood disorder
    'Anxiety disorders': ('anxiety disorder', 'http://purl.obolibrary.org/obo/DOID_150'),  # disease of mental health
    'Bipolar disorders': ('bipolar disorder', 'http://purl.obolibrary.org/obo/DOID_3324'),  # mood disorder
    'Eating disorders': ('eating disorder', 'http://purl.obolibrary.org/obo/DOID_0060037')  # specific developmental disorder
}

# Creazione delle colonne URI nel DataFrame
for col, (label, parent_class) in ontology_disorders.items():
    uri_col = f'{col}_URI'
    df[uri_col] = df[col].apply(lambda x: get_disorder_uri(label, parent_class))

# Salva il dataset arricchito
output_path = r'ICON-20240618-0005\Europa\Risultati\dataset_arricchito.csv'
df.to_csv(output_path, index=False)

# Visualizza il dataset arricchito per assicurarsi che le URI siano state aggiunte correttamente
print(df.head())
