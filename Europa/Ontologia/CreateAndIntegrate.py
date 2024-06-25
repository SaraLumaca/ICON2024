import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD, OWL

# Load the dataset
file_path = 'Europa/Risultati/DisturbiMentali-DalysNazioniDelMondo-GruppoDiIntervento.csv'
df = pd.read_csv(file_path)

# Define namespaces
FUTURA = Namespace("http://futuramente.org/ontologies/2023#")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# Mapping of disorders to their URIs
disorder_uris = {
    "Schizophrenia": OBO.DOID_5419,
    "Depressive": OBO.DOID_1596,
    "Anxiety": OBO.DOID_2030,
    "Bipolar": OBO.DOID_3312,
    "Eating": OBO.DOID_8670
}

# Create a new graph for the integrated ontology
g = Graph()

# Bind namespaces
g.bind("futura", FUTURA)
g.bind("obo", OBO)

# Import the existing ontology
existing_ontology_path = 'Europa/Ontologia/HumanDiseaseOntology.owl'
g.parse(existing_ontology_path)

# Function to create RDF triples
def create_rdf_triples(row):
    country = URIRef(FUTURA + row['Entity'].replace(" ", "_"))
    year = Literal(row['Year'], datatype=XSD.gYear)
    group = row['Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")']
    
    g.add((country, RDF.type, FUTURA.Country))
    g.add((country, FUTURA.hasYear, year))
    
    for disorder, uri in disorder_uris.items():
        g.add((country, FUTURA.hasDisorder, uri))
    
    if group == 0:
        group_label = FUTURA.EconomicDevelopment
    elif group == 1:
        group_label = FUTURA.HighIncome
    else:
        group_label = FUTURA.LowIncome
    
    g.add((country, FUTURA.belongsToGroup, group_label))

    # Add other columns as properties
    g.add((country, FUTURA.schizophreniaDisorders, Literal(row['Schizophrenia disorders'])))
    g.add((country, FUTURA.depressiveDisorders, Literal(row['Depressive disorders'])))
    g.add((country, FUTURA.anxietyDisorders, Literal(row['Anxiety disorders'])))
    g.add((country, FUTURA.bipolarDisorders, Literal(row['Bipolar disorders'])))
    g.add((country, FUTURA.eatingDisorders, Literal(row['Eating disorders'])))
    g.add((country, FUTURA.dalysDepressiveDisorders, Literal(row['DALYs Cause: Depressive disorders'])))
    g.add((country, FUTURA.dalysSchizophrenia, Literal(row['DALYs Cause: Schizophrenia'])))
    g.add((country, FUTURA.dalysBipolarDisorder, Literal(row['DALYs Cause: Bipolar disorder'])))
    g.add((country, FUTURA.dalysEatingDisorders, Literal(row['DALYs Cause: Eating disorders'])))
    g.add((country, FUTURA.dalysAnxietyDisorders, Literal(row['DALYs Cause: Anxiety disorders'])))
    g.add((country, FUTURA.clusterKMeans, Literal(row['Cluster_KMeans'])))

# Create RDF triples for all rows in the dataset
df.apply(create_rdf_triples, axis=1)

# Serialize the graph to an OWL file
output_path = r'Europa/Risultati/IntegratedOntology.owl'
g.serialize(destination=output_path, format='xml')
