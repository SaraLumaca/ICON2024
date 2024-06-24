import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn import preprocessing, metrics
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# Lettura dei dati da file CSV
Data1 = pd.read_csv(r"DisturbiMentali-DalysNazioniDelMondo.csv")
Data2 = pd.read_csv(r'4- adult-population-covered-in-primary-data-on-the-prevalence-of-mental-illnesses.csv')
Data3 = pd.read_csv(r'6- depressive-symptoms-across-us-population.csv')
Data4 = pd.read_csv(r'7- number-of-countries-with-primary-data-on-prevalence-of-mental-illnesses-in-the-global-burden-of-disease-study.csv')

# Creazione dei DataFrame
df1 = pd.DataFrame(Data1)
df2 = pd.DataFrame(Data2)
df3 = pd.DataFrame(Data3)
df4 = pd.DataFrame(Data4)

# Funzione per generare una tabella di descrizione dei dati
# mettere nel preprocessing
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

# Funzioni per visualizzare i grafici
def bar_chart_major_depression():
    df2.sort_values(by="Major depression", inplace=True)
    fig = px.bar(df2, x="Major depression", y="Entity", orientation='h', color='Bipolar disorder')
    fig.show()

def bar_chart_eating_disorders():
    df2.sort_values(by="Eating disorders", inplace=True)
    fig = px.bar(df2, x="Eating disorders", y="Entity", orientation='h', color='Dysthymia')
    fig.show()

def bar_chart_schizophrenia():
    df2.replace(to_replace="<0.1", value=0.1, regex=True, inplace=True)
    df2['Schizophrenia'] = df2['Schizophrenia'].astype(float)
    df2.sort_values(by="Schizophrenia", inplace=True)
    fig = px.bar(df2, x="Schizophrenia", y="Entity", orientation='h', color='Anxiety disorders')
    fig.show()

def correlation_heatmap():
    df1.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }, inplace=True)
    df1_variables = df1[["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders", "Eating disorders"]]
    Corrmat = df1_variables.corr()
    plt.figure(figsize=(10, 8), dpi=200)
    sns.heatmap(Corrmat, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione per le Malattie Mentali')
    plt.show()

def scatter_plot():
    df1.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }, inplace=True)
    df1_variables = df1[["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders", "Eating disorders"]]
    diseases = ["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders", "Eating disorders"]
    
    print("Available diseases for scatter plot:")
    for i, disease in enumerate(diseases, 1):
        print(f"{i}. {disease}")

    disease1_idx = int(input("Select the first disease (number): ")) - 1
    disease2_idx = int(input("Select the second disease (number): ")) - 1
    
    plt.figure(figsize=(10, 6), dpi=200)
    sns.scatterplot(x=diseases[disease1_idx], y=diseases[disease2_idx], data=df1_variables)
    plt.title(f'{diseases[disease1_idx]} vs {diseases[disease2_idx]}')
    plt.xlabel(diseases[disease1_idx])
    plt.ylabel(diseases[disease2_idx])
    plt.show()

def subplot_major_bipolar():
    fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)
    x1 = ["Andean Latin America", "West Sub-Saharan Africa", "Tropical Latin America", "Central Asia", "Central Europe",
          "Central Sub-Saharan Africa", "Southern Latin America", "North Africa/Middle East", "Southern Sub-Saharan Africa",
          "Southeast Asia", "Oceania", "Central Latin America", "Eastern Europe", "South Asia", "East Sub-Saharan Africa",
          "Western Europe", "World", "East Asia", "Caribbean", "Asia Pacific", "Australasia", "North America"]

    fig.append_trace(go.Bar(x=df2["Bipolar disorder"], y=x1, marker=dict(color='rgba(50, 171, 96, 0.6)',
                     line=dict(color='rgba(20, 10, 56, 1.0)', width=0)), name='Bipolar disorder in Mental Health', orientation='h'), 1, 1)
    
    fig.append_trace(go.Scatter(x=df2["Major depression"], y=x1, mode='lines+markers', line_color='rgb(40, 0, 128)', name='Major depression in Mental Health'), 1, 2)

    fig.update_layout(
        title='Major depression and Bipolar disorder',
        yaxis=dict(showgrid=False, showline=False, showticklabels=True, domain=[0, 0.85]),
        yaxis2=dict(showgrid=False, showline=True, showticklabels=False, linecolor='rgba(102, 102, 102, 0.8)', linewidth=5, domain=[0, 0.85]),
        xaxis=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0, 0.45]),
        xaxis2=dict(zeroline=False, showline=False, showticklabels=True, showgrid=True, domain=[0.47, 1], side='top', dtick=10000),
        legend=dict(x=0.029, y=1.038, font_size=10),
        margin=dict(l=100, r=20, t=70, b=70),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        annotations=[dict(xref='x2', yref='y2', y=xd, x=ydn+10, text='{:,}'.format(ydn) + '%', font=dict(family='Arial', size=10, color='rgb(128, 0, 128)'), showarrow=False)
                     for ydn, yd, xd in zip(df2["Major depression"], df2["Bipolar disorder"], x1)] +
                     [dict(xref='x1', yref='y1', y=xd, x=yd+10, text=str(yd) + '%', font=dict(family='Arial', size=10, color='rgb(50, 171, 96)'), showarrow=False)
                     for ydn, yd, xd in zip(df2["Major depression"], df2["Bipolar disorder"], x1)] +
                     [dict(xref='paper', yref='paper', x=-0.2, y=-0.109, text="Visualizzazione della salute mentale", font=dict(family='Arial', size=20, color='rgb(150,150,150)'), showarrow=False)]
    )
    fig.show()

def line_chart_depressive_symptoms():
    x = ["Appetite change", "Average across symptoms", "Depressed mood", "Difficulty concentrating", "Loss of interest",
         "Low energy", "Low self-esteem", "Psychomotor agitation", "Psychomotor agitation", "Sleep problems", "Suicidal ideation"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df3["Nearly every day"], name='Nearly every day', line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df3["More than half the days"], name='More than half the days', line=dict(color='royalblue', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df3["Several days"], name='Several days', line=dict(color='black', width=4, dash='dashdot')))
    fig.update_layout(title='Depressive symptoms across us population', xaxis_title='Entity', yaxis_title='Types of days')
    fig.show()

def line_chart_mental_illness():
    x = ["Alcohol use disorders", "Amphetamine use disorders", "Anorexia nervosa", "Anxiety disorders",
         "Attention-deficit hyperactivity disorder", "Autism spectrum disorders", "Bipolar disorder",
         "Bulimia nervosa", "Cannabis use disorders", "Cocaine use disorders", "Dysthymia","Major depressive disorder",
         "Opioid use disorders", "Other drug use disorders", "Personality disorders"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=df4["Number of countries with primary data on prevalence of mental disorders"], name='Nearly every day', line=dict(color='firebrick', width=4)))
    fig.update_layout(title='Malattie mentali nello studio del carico globale di malattia', xaxis_title='Malattie', yaxis_title='Numero di paesi')
    fig.show()

def box_plots():
    df1.rename(columns={
        'Schizophrenia disorders (share of population) - Sex: Both - Age: Age-standardized': 'Schizophrenia disorders',
        'Depressive disorders (share of population) - Sex: Both - Age: Age-standardized': 'Depressive disorders',
        'Anxiety disorders (share of population) - Sex: Both - Age: Age-standardized': 'Anxiety disorders',
        'Bipolar disorders (share of population) - Sex: Both - Age: Age-standardized': 'Bipolar disorders',
        'Eating disorders (share of population) - Sex: Both - Age: Age-standardized': 'Eating disorders'
    }, inplace=True)
    df1_variables = df1[["Schizophrenia disorders", "Depressive disorders", "Anxiety disorders", "Bipolar disorders", "Eating disorders"]]
    Numerical = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders', 'Eating disorders']
    fig = make_subplots(rows=1, cols=5, subplot_titles=Numerical)
    for i in range(5):
        trace = go.Box(x=df1_variables[Numerical[i]], name=Numerical[i])
        fig.add_trace(trace, row=1, col=i+1)
    fig.update_layout(height=300, width=1200, title_text="Boxplots")
    fig.update_layout(showlegend=False)
    fig.show()

# Funzione per eseguire la regressione lineare e visualizzare i risultati
def linear_regression_analysis(dependent_variable):
    # Selezione delle colonne necessarie per l'analisi
    features = ['Schizophrenia disorders', 'Depressive disorders', 'Anxiety disorders', 'Bipolar disorders', 'Eating disorders']
    features.remove(dependent_variable)
    
    X_model = df1[features]
    y_model = df1[dependent_variable]
    
    # Stampa di debug
    print(f"\nPredicting {dependent_variable} using {features}\n")
    
    # Normalizzazione delle features
    scaler = preprocessing.MinMaxScaler()
    X_model_norm = scaler.fit_transform(X_model)
    
    # Divisione del dataset in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X_model_norm, y_model, test_size=0.2, random_state=42)
    print("Shape of X_train: ", X_train.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of X_test: ", X_test.shape)
    print("Shape of y_test: ", y_test.shape)
    
    # Creazione e addestramento del modello di regressione lineare
    Model = LinearRegression()
    Model.fit(X_train, y_train)
    
    # Predizione sui dati di test
    y_pred = Model.predict(X_test)
    
    # Stampa dei risultati
    print("Coefficients:")
    for feature, coef in zip(features, Model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"Intercept: {Model.intercept_:.4f}\n")
    
    print("Mean Absolute Error (MAE): ", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error (MSE): ", metrics.mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error (RMSE): ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print("R2 Score: ", metrics.r2_score(y_test, y_pred))
    
    # Validazione incrociata a 10 fold
    k_fold = KFold(10)
    cv_scores = cross_val_score(Model, X_model_norm, y_model.ravel(), cv=k_fold, n_jobs=1)
    print("Cross-validation scores: ", cv_scores)
    
    # Funzione per verificare l'aggiunta di nuove feature con una certa dimensione
    #def check(X_train, X_test, y_train, y_test, y_pred, r2, Dimension, testsize):
    #    for column in range(X_train.shape[1]):
    #        New_Col_name = f"X{column}^{Dimension}"
    #        New_Col_value = X_train[:, column] ** Dimension
    #        X_train = np.column_stack((New_Col_value, X_train))
    #        New_Col_value_test = X_test[:, column] ** Dimension
    #        X_test = np.column_stack((New_Col_value_test, X_test))
    #        New_model = LinearRegression()
    #        New_model.fit(X_train, y_train)
    #        y_pred = New_model.predict(X_test)
    #        r2_new = metrics.r2_score(y_test, y_pred)
    #        if r2_new < r2:
    #            X_train = X_train[:, 1:]
    #            X_test = X_test[:, 1:]
    #        else:
    #            r2 = r2_new
    #    print("R2 score with polynomial features: ", r2)
    
    # Aggiunta di nuove feature quadrate e verifica
    #r2_initial = metrics.r2_score(y_test, y_pred)
    #check(X_train, X_test, y_train, y_test, y_pred, r2_initial, 2, 0.2)
    
    # Creazione di feature interattive e aggiunta al DataFrame
    #X_model['Bipolar_Anx'] = X_model["Bipolar disorders"] * X_model["Anxiety disorders"]
    #X_model['Dep_Schi'] = X_model["Depressive disorders"] * X_model["Schizophrenia disorders"]

    
    # Divisione dei dati in training e test set con nuove feature
    #X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=0)
    
    # Creazione di un nuovo modello e addestramento con le nuove feature
    #Model = LinearRegression()
    #Model.fit(X_train, y_train)
    #y_pred = Model.predict(X_test)
    #r2 = metrics.r2_score(y_test, y_pred)
    #print("R2 Score with additional features: ", r2)
    
    # Funzione per visualizzare le predizioni
    def plot_predictions(a, b, c, d, title, xlabel, ylabel, real_label, pred_label, real_color, pred_color):
        font1 = {'family':'fantasy','color':'blue','size':20}
        font2 = {'family':'serif','color':'darkred','size':20}
        font3 = {'family':'cursive','color':'green','size':20}
        plt.figure(figsize=(20,10), dpi=200)
        plt.title(title, fontdict=font2)
        plt.xlabel(xlabel, fontdict=font3)
        plt.ylabel(ylabel, fontdict=font1)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
    
        # Plotting scatter points
        plt.scatter(a, b, color=real_color, label=real_label, alpha=0.6, s=50)
        plt.scatter(c, d, color=pred_color, label=pred_label, marker="H", s=80, alpha=0.6)
    
        # Fitting and plotting the trend lines for real values
        model_real = LinearRegression().fit(np.array(a).reshape(-1, 1), b)
        trendline_real = model_real.predict(np.array(a).reshape(-1, 1))
        plt.plot(a, trendline_real, color=real_color, linestyle='dashed', linewidth=3, label=f'Trend Line ({real_label})')
    
        # Fitting and plotting the trend lines for predicted values
        model_pred = LinearRegression().fit(np.array(c).reshape(-1, 1), d)
        trendline_pred = model_pred.predict(np.array(c).reshape(-1, 1))
        plt.plot(c, trendline_pred, color=pred_color, linestyle='dashed', linewidth=3, label=f'Trend Line ({pred_label})')
    
        plt.legend(fontsize=15)
        plt.grid(True)
        plt.show()

    # Visualizzazione delle predizioni per ogni variabile indipendente
        for i, feature in enumerate(features):
           plot_predictions(X_test[:, i], y_test, X_test[:, i], y_pred,
                         f"{feature} Prediction with Trend Lines", feature, dependent_variable,
                         "Real Values", "Predicted Values", 'blue', 'red')
    
    # Plotting scatter points
        plt.scatter(a, b, color=real_color, label=real_label, alpha=0.6, s=50)
        plt.scatter(c, d, color=pred_color, label=pred_label, marker="H", s=80, alpha=0.6)
    
    # Fitting and plotting the trend lines for real values
        model_real = LinearRegression().fit(np.array(a).reshape(-1, 1), b)
        trendline_real = model_real.predict(np.array(a).reshape(-1, 1))
        plt.plot(a, trendline_real, color=real_color, linestyle='dashed', linewidth=3, label=f'Trend Line ({real_label})')
    
    # Fitting and plotting the trend lines for predicted values
        model_pred = LinearRegression().fit(np.array(c).reshape(-1, 1), d)
        trendline_pred = model_pred.predict(np.array(c).reshape(-1, 1))
        plt.plot(c, trendline_pred, color=pred_color, linestyle='dashed', linewidth=3, label=f'Trend Line ({pred_label})')
    
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()


    #plot_predictions(X_test["Bipolar disorders"], y_test, X_test["Bipolar disorders"], y_pred,
    #             "Bipolar Prediction with Trend Lines", "Bipolar", "Eating", "Real Values", "Predicted Values", 'blue', 'red')

   # plot_predictions(X_test["Schizophrenia disorders"], y_test, X_test["Schizophrenia disorders"], y_pred,
    #             "Schizophrenia Prediction with Trend Lines", "Schizophrenia", "Eating", "Real Values", "Predicted Values", 'blue', 'orange')

    #plot_predictions(X_test["Anxiety disorders"], y_test, X_test["Anxiety disorders"], y_pred,
    #             "Anxiety Prediction with Trend Lines", "Anxiety", "Eating", "Real Values", "Predicted Values", 'blue', 'indigo')

    #plot_predictions(X_test["Depressive disorders"], y_test, X_test["Depressive disorders"], y_pred,
    #             "Depressive Prediction with Trend Lines", "Depressive", "Eating", "Real Values", "Predicted Values", 'blue', 'green')


def main_menu():
    while True:
        print("\nMenu Principale:")
        print("1. Analisi Descrittiva del Dataset")
        print("2. Elaborazione dei Dati")
        print("0. Esci")
        
        main_choice = input("Inserisci la tua scelta: ")
        
        if main_choice == "1":
            while True:
                print("\nAnalisi Descrittiva del Dataset:")
                print("1. Grafico a Barre: Depressione Maggiore")
                print("2. Grafico a Barre: Disturbi Alimentari")
                print("3. Grafico a Barre: Schizofrenia")
                print("4. Heatmap di Correlazione")
                print("5. Grafico a Dispersione")
                print("6. Subplot: Depressione Maggiore e Disturbo Bipolare")
                print("7. Grafico a Linee: Sintomi Depressivi")
                print("8. Grafico a Linee: Malattie Mentali")
                print("9. Box Plot")
                print("0. Torna al Menu Principale")
                
                choice = input("Inserisci la tua scelta: ")
                
                if choice == "1":
                    bar_chart_major_depression()
                elif choice == "2":
                    bar_chart_eating_disorders()
                elif choice == "3":
                    bar_chart_schizophrenia()
                elif choice == "4":
                    correlation_heatmap()
                elif choice == "5":
                    scatter_plot()
                elif choice == "6":
                    subplot_major_bipolar()
                elif choice == "7":
                    line_chart_depressive_symptoms()
                elif choice == "8":
                    line_chart_mental_illness()
                elif choice == "9":
                    box_plots()
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida. Per favore, riprova.")
        
        elif main_choice == "2":
            while True:
                print("\nElaborazione dei Dati:")
                print("1. Analisi di Regressione Lineare")
                print("0. Torna al Menu Principale")
                
                choice = input("Inserisci la tua scelta: ")

                if choice == "1":
                    diseases = [
                         'Schizophrenia disorders',
                         'Depressive disorders',
                         'Anxiety disorders',
                         'Bipolar disorders',
                         'Eating disorders'
                        ]
                    # Eseguire la regressione lineare per ogni patologia
                    for disease in diseases:
                        linear_regression_analysis(disease)
                elif choice == "0":
                    break
                else:
                    print("Scelta non valida. Per favore, riprova.")
        
        elif main_choice == "0":
            break
        else:
            print("Scelta non valida. Per favore, riprova.")

if __name__ == "__main__":
    main_menu()
