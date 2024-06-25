import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn import metrics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot  # Import plot instead of init_notebook_mode

import warnings
warnings.filterwarnings("ignore")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot

#Lettura dei dati da file CSV
Data1 = pd.read_csv(r"DisturbiMentali-DalysNazioniDelMondo-GruppoDiIntervento.csv")
Data2= pd.read_csv(r'4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv')
Data3= pd.read_csv(r'6- depressive-symptoms-across-us-population.csv')
Data4 = pd.read_csv(r'7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv')

#Creazione dei DataFrame
df1 = pd.DataFrame(Data1)
df2 = pd.DataFrame(Data2)
df3 = pd.DataFrame(Data3)
df4 = pd.DataFrame(Data4)

#Funzione per generare una tabella di descrizione dei dati
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

#Classe per la gestione dei colori nei messaggi di stampa
class color:
   BLUE = '\033[94m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

#Classe delle tabelle di descrizione per i quattro DataFrame
print(color.BOLD + color.BLUE + color.UNDERLINE +
      '"The describe table of df1 : Mental illness dataframe"' + color.END)
print(describe(df1))
print("\n")
print(color.BOLD + color.BLUE + color.UNDERLINE +
      '"The describe table of df2 : Adult population, mental illnesses"' + color.END)
print(describe(df2))
print("\n")
print(color.BOLD + color.BLUE + color.UNDERLINE +
      '"The describe table of df3 : Depressive"' + color.END)
print(describe(df3))
print("\n")
print(color.BOLD + color.BLUE + color.UNDERLINE +
      '"The describe table of df4 : Number of countries"' + color.END)
print(describe(df4))

#Visualizzazione di grafici con Plotply
df2.sort_values(by= "Major depression" ,inplace=True)
plt.figure(dpi=200)
fig = px.bar(df2, x="Major depression", y="Entity", orientation='h',color='Bipolar disorder')
fig.show()

# Ordina il DataFrame df2 in base alla colonna "Eating disorders
df2.sort_values(by= "Eating disorders" ,inplace=True)

# Crea un grafico a barre orizzontali con Plotly Express
plt.figure(dpi=200)
fig = px.bar(df2, x="Eating disorders", y="Entity", orientation='h',color='Dysthymia')
fig.show()

# Sostituisci i valori "<0.1" con 0.1 nel DataFrame df2
df2.replace(to_replace="<0.1", value=0.1, regex=True, inplace=True)

# Converti la colonna 'Schizophrenia' in float nel DataFrame df2
df2['Schizophrenia'] = df2['Schizophrenia'].astype(float)


# Ordina il DataFrame df2 in base alla colonna "Schizophrenia"
df2.sort_values(by= "Schizophrenia" ,inplace=True)

# Crea un secondo grafico a barre orizzontali con Plotly Express
plt.figure(dpi=200)
fig = px.bar(df2, x="Schizophrenia", y="Entity", orientation='h',color='Anxiety disorders')
fig.show()

# Crea un subplot con due grafici: uno a barre e uno a linee
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

# Definisci l'elenco delle entità per l'asse y
x1 = ["Andean Latin America", "West Sub-Saharan Africa", "Tropical Latin America", "Central Asia", "Central Europe",
    "Central Sub-Saharan Africa", "Southern Latin America", "North Africa/Middle East", "Southern Sub-Saharan Africa",
    "Southeast Asia", "Oceania", "Central Latin America", "Eastern Europe", "South Asia", "East Sub-Saharan Africa",
    "Western Europe", "World", "East Asia", "Caribbean", "Asia Pacific", "Australasia", "North America"]

# Aggiungi un grafico a barre al primo subplot
fig.append_trace(go.Bar(
    x=df2["Bipolar disorder"],
    y=x1,
    marker=dict(
        color='rgba(50, 171, 96, 0.6)',
        line=dict(
            color='rgba(20, 10, 56, 1.0)',
            width=0),
    ),
    name='Bipolar disorder in Mental Health',
    orientation='h',
), 1, 1)

# Aggiungi un grafico a linee e marker al secondo subplot
fig.append_trace(go.Scatter(
    x=df2["Major depression"], y=x1,
    mode='lines+markers',
    line_color='rgb(40, 0, 128)',
    name='Major depression in Mental Health',
), 1, 2)


# Aggiorna il layout del grafico
fig.update_layout(
    title='Major depression and Bipolar disorder',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=True,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=5,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.45],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        side='top',
        dtick=10000,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

# Aggiungi etichette al grafico
annotations = []


# Aggiungi etichette per i dati di depressione maggiore e disturbo bipolare
for ydn, yd, xd in zip(df2["Major depression"], df2["Bipolar disorder"], x1):
    # Etichetta per i dati di depressione maggiore
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn+10,
                            text='{:,}'.format(ydn) + '%',
                            font=dict(family='Arial', size=10,
                                      color='rgb(128, 0, 128)'),
                            showarrow=False))
    # Etichetta per i dati di disturbo bipolare
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd+10,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=10,
                                      color='rgb(50, 171, 96)'),
                            showarrow=False))

# Aggiungi etichetta della fonte
annotations.append(dict(xref='paper', yref='paper',
                        x=-0.2, y=-0.109,
                        text="Visualizzazione della salute mentale",
                        font=dict(family='Arial', size=20, color='rgb(150,150,150)'),
                        showarrow=False))

# Aggiorna il layout con le annotazioni
fig.update_layout(annotations=annotations)

# Visualizza il grafico
fig.show()

# Definisci le categorie per l'asse x
x = ["Appetite change", "Average across symptoms", "Depressed mood", "Difficulty concentrating", "Loss of interest",
    "Low energy", "Low self-esteem", "Psychomotor agitation", "Psychomotor agitation", "Sleep problems", "Suicidal ideation"]

# Crea un grafico a linee per i sintomi depressivi
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=df3["Nearly every day"], name='Nearly every day',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=x, y=df3["More than half the days"], name='More than half the days',
                         line=dict(color='royalblue', width=4)))
fig.add_trace(go.Scatter(x=x, y=df3["Several days"], name='Several days',
                         line=dict(color='black', width=4,
                              dash='dashdot') # dash options include 'dash', 'dot', and 'dashdot'
))

# Edit the layout
fig.update_layout(title='Depressive symptoms across us population',
                   xaxis_title='Entity',
                   yaxis_title='Types of days')


fig.show()

x = ["Alcohol use disorders", "Amphetamine use disorders", "Anorexia nervosa", "Anxiety disorders",
     "Attention-deficit hyperactivity disorder", "Autism spectrum disorders", "Bipolar disorder",
     "Bulimia nervosa", "Cannabis use disorders", "Cocaine use disorders", "Dysthymia","Major depressive disorder",
    "Opioid use disorders", "Other drug use disorders", "Personality disorders"]

fig = go.Figure()
# Create and style traces
fig.add_trace(go.Scatter(x=x, y=df4["Number of countries with primary data on prevalence of mental disorders"],
                         name='Nearly every day',
                         line=dict(color='firebrick', width=4)))


# Modifica il layout del primo grafico
fig.update_layout(title='Sintomi depressivi nella popolazione statunitense',
                   xaxis_title='Entità',
                   yaxis_title='Tipi di giorni')

# Visualizza il primo grafico
fig.show()

# Definisci le categorie per l'asse x del secondo grafico
x = ["Alcohol use disorders", "Amphetamine use disorders", "Anorexia nervosa", "Anxiety disorders",
     "Attention-deficit hyperactivity disorder", "Autism spectrum disorders", "Bipolar disorder",
     "Bulimia nervosa", "Cannabis use disorders", "Cocaine use disorders", "Dysthymia","Major depressive disorder",
    "Opioid use disorders", "Other drug use disorders", "Personality disorders"]

# Crea un grafico a linee per le malattie mentali nello studio del carico globale di malattia
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=df4["Number of countries with primary data on prevalence of mental disorders"],
                         name='Nearly every day',
                         line=dict(color='firebrick', width=4)))

# Modifica il layout del secondo grafico
fig.update_layout(title='Malattie mentali nello studio del carico globale di malattia',
                   xaxis_title='Malattie',
                   yaxis_title='Numero di paesi')

# Visualizza il secondo grafico
fig.show()

# Ottieni i nomi delle colonne del DataFrame df1
df1_column_names = list(df1.columns.values)

# Rinomina le colonne del DataFrame df1
df1 = df1.rename(columns={'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders', 
                          'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
                         'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized':'Anxiety disorders',
                         'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized':'Bipolar disorders',
                         'Eating disorders (share of population) - Sex: Both - Age: Age-standardized':'Eating disorders'})

# Seleziona le colonne rilevanti nel DataFrame df1
df1_variables = df1[["Schizophrenia disorders","Depressive disorders","Anxiety disorders","Bipolar disorders",
                       "Eating disorders"]]

# Calcola la matrice di correlazione tra le variabili
Corrmat = df1_variables.corr()

# Crea un grafico a dispersione tra Schizophrenia ed Eating disorders
plt.figure(figsize=(15, 10), dpi=200)
plt.title('Schizophrenia - Eating')
sns.scatterplot(x="Schizophrenia disorders", y="Eating disorders", data=df1_variables)

# Crea un grafico a dispersione tra Depressive ed Eating disorders
plt.figure(figsize=(15, 10), dpi=200)
plt.title('Depressive - Eating')
sns.scatterplot(x='Depressive disorders', y="Eating disorders", data=df1_variables)

# Crea un grafico a dispersione tra Anxiety ed Eating disorders
plt.figure(figsize=(15, 10), dpi=200)
plt.title('Anxiety - Eating')
sns.scatterplot(x='Anxiety disorders', y="Eating disorders", data=df1_variables)

# Crea un grafico a dispersione tra Bipolar ed Eating disorders
plt.figure(figsize=(15, 10), dpi=200)
plt.title('Bipolar - Eating')
sns.scatterplot(x='Bipolar disorders', y="Eating disorders", data=df1_variables)

# Crea boxplots per le variabili numeriche nel DataFrame df1
Numerical = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders', 'Eating disorders']
fig = make_subplots(rows=1, cols=5, subplot_titles=Numerical)

for i in range(5):
    trace = go.Box(x=df1_variables[Numerical[i]], name=Numerical[i])
    fig.add_trace(trace, row=1, col=i+1)

# Modifica il layout del grafico a boxplots
fig.update_layout(height=300, width=1200, title_text="Boxplots")
fig.update_layout(showlegend=False)  # Nascondi la legenda se non è necessaria

# Visualizza il grafico a boxplots
plot(fig)

# Seleziona le colonne necessarie per l'analisi di regressione lineare
features = ['Schizophrenia disorders', 'Depressive disorders','Anxiety disorders','Bipolar disorders']
X_model = df1[features]
y_model = df1["Eating disorders"]

# Stampa di debug dataframe
print("Columns in df1:", df1.columns)
print("Columns in X_model:", X_model.columns)

# Normalizza le features utilizzando MinMaxScaler
scaler = preprocessing.MinMaxScaler()
X_model_norm = scaler.fit_transform(X_model)

# Divisione del dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X_model_norm, y_model, test_size=20, random_state=42)
print("Shape of x_train : ", X_train.shape)
print("Shape of y_train : ", y_train.shape)
print("Shape of x_test : ", X_test.shape)
print("Shape of y_test : ", y_test.shape)

# Creazione di un modello di regressione lineare e addestramento
Model = LinearRegression()
Model.fit(X_train, y_train)

# Predizione sui dati di test
y_pred = Model.predict(X_test)
print("Mean Absolute Error of Model is: ", metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error of Model is: ", metrics.mean_squared_error(y_test,y_pred))
print("Root Mean Squared of Model is: ", np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("R2 Score of Model is: ", metrics.r2_score(y_test,y_pred))

# Validazione incrociata a 10 fold
k_fold = KFold(10)
print (cross_val_score(Model, X_model_norm, y_model.ravel(), cv=k_fold, n_jobs=1))

# Funzione per verificare l'aggiunta di nuove feature con una certa dimensione
def check(Dimension, testsize):
    r2 = 0.6289
    for column in X_model:
        New_Col_name = column + str(Dimension)
        New_Col_value = X_model[column]**Dimension
        X_model.insert(0, New_Col_name, New_Col_value)
        X_train, X_test, y_train, y_test = train_test_split(X_model, y_model,test_size=testsize,random_state=0)
        New_model = LinearRegression()
        New_model.fit(X_train, y_train)
        y_pred = New_model.predict (X_test)
        r2_new = metrics.r2_score(y_test, y_pred)
        if r2_new < r2:
            X_model.drop([New_Col_name], axis=1, inplace=True)
        else:
            r2 = r2_new
            
    print("R2 score is: ", r2)

# Aggiunta di nuove feature quadrate e verifica
check(2,0.2)

# Creazione di feature interattive e aggiunta al DataFrame
X_model['Bipolar_Anx'] = X_model["Bipolar disorders"] * X_model["Anxiety disorders"]
#X_model['Bipolar_Anx2'] = X_model["Bipolar disorders2"] * X_model["Anxiety disorders2"]
X_model['Dep_Schi'] = X_model["Depressive disorders"] * X_model["Schizophrenia disorders"]
#X_model['Dep_Schi2'] = X_model["Depressive disorders2"] * X_model["Schizophrenia disorders2"]

# Divisione dei dati in training e test set con nuove feature
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=0)

# Creazione di un nuovo modello e addestramento con le nuove feature
Model = LinearRegression()
Model.fit(X_train, y_train)
y_pred = Model.predict(X_test)
r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score with additional features: ", r2)

# Plot delle predizioni contro i valori reali per la feature Bipolar
a = X_test["Bipolar disorders"]
b = y_test
c = X_test["Bipolar disorders"]
d = y_pred

font1 = {'family':'fantasy','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':20}
font3 = {'family':'cursive','color':'green','size':20}
plt.figure(figsize= (20,10), dpi=200)
plt.title("Bipolar Prediction",fontdict=font2)
plt.xlabel("Bipolar",fontdict= font3)
plt.ylabel("Eating",fontdict=font1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(a,b, color = 'blue', label = "Real Values")
plt.scatter(c,d, color = 'maroon', label = "Predicted Values", marker="H", s=80)
plt.legend(fontsize=15)
plt.show()

# Plot delle predizioni contro i valori reali per la feature Schizophrenia
a1 = X_test["Schizophrenia disorders"]
b1 = y_test
c1 = X_test["Schizophrenia disorders"]
d1 = y_pred

plt.figure(figsize= (20,10), dpi=200)
plt.title("Schizophrenia Prediction",fontdict=font2)
plt.xlabel("Schizophrenia",fontdict= font3)
plt.ylabel("Eating",fontdict=font1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(a1,b1, color = 'blue', label = "Real Values")
plt.scatter(c1,d1, color = 'Orange', label = "Predicted Values", marker="H", s=80)
plt.legend(fontsize=15)
plt.show()

# Plot delle predizioni contro i valori reali per la feature Anxiety
a2 = X_test["Anxiety disorders"]
b2 = y_test
c2 = X_test["Anxiety disorders"]
d2 = y_pred

plt.figure(figsize= (20,10), dpi=200)
plt.title("Anxiety Prediction",fontdict=font2)
plt.xlabel("Anxiety",fontdict= font3)
plt.ylabel("Eating",fontdict=font1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(a2,b2, color = 'blue', label = "Real Values")
plt.scatter(c2,d2, color = 'indigo', label = "Predicted Values", marker="H", s=80)
plt.legend(fontsize=15)
plt.show()

# Plot delle predizioni contro i valori reali per la feature Depressive
a3 = X_test["Depressive disorders"]
b3 = y_test
c3 = X_test["Depressive disorders"]
d3 = y_pred

plt.figure(figsize= (20,10), dpi=200)
plt.title("Depressive Prediction",fontdict=font2)
plt.xlabel("Depressive",fontdict= font3)
plt.ylabel("Eating",fontdict=font1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.scatter(a3,b3, color = 'blue', label = "Real Values")
plt.scatter(c3,d3, color = 'green', label = "Predicted Values", marker="H", s=80)
plt.legend(fontsize=15)
plt.show()
