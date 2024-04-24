# data_visualization.py

# Il modulo `data_visualization.py` fornisce funzioni per la visualizzazione dei dati e grafici

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# 1. `plot_bar_chart(df, x_column, y_column, color_column=None)`: Questa funzione crea un grafico a barre utilizzando Plotly Express. Prende in input un DataFrame pandas (`df`), il nome della colonna da utilizzare sull'asse x (`x_column`), il nome della colonna da utilizzare sull'asse y (`y_column`) e, opzionalmente, il nome della colonna per la colorazione delle barre (`color_column`). Mostra il grafico interattivo.
def plot_bar_chart(df, x_column, y_column, color_column=None):
    fig = px.bar(df, x=x_column, y=y_column, color=color_column)
    fig.show()

#1.a


def plot_bar_line_chart(df, x_column, y1_column, y2_column):
    fig, ax1 = plt.subplots()

    # Formattazione dei dati in ingresso
    x_data = df[x_column]
    y1_data = df[y1_column]
    y2_data = df[y2_column]

    ax1.bar(range(len(x_data)), y1_data, color='b', alpha=0.5, label=y1_column)
    ax1.set_xlabel(x_column)
    ax1.set_ylabel(y1_column, color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(range(len(x_data)), y2_data, color='r', label=y2_column)
    ax2.set_ylabel(y2_column, color='r')
    ax2.tick_params('y', colors='r')

    # Impostazione del numero fisso di ticks sull'asse x
    ax1.set_xticks(range(len(x_data)))
    ax1.set_xticklabels(x_data, rotation=90)

    fig.tight_layout()
    plt.show()




# 2. `plot_scatter_chart(x, y, title, x_label, y_label)`: Questa funzione crea un grafico a dispersione utilizzando Plotly Graph Objects. Prende in input i dati sull'asse x (`x`) e sull'asse y (`y`), il titolo del grafico (`title`), le etichette degli assi x e y (`x_label` e `y_label`). Mostra il grafico interattivo.
def plot_scatter_chart(x, y, title, x_label, y_label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    fig.show()

# 3. `plot_box_plots(df, columns)`: Questa funzione crea dei grafici a boxplot utilizzando Plotly Express. Prende in input un DataFrame pandas (`df`) e una lista di nomi di colonne (`columns`) per le quali si desidera generare i boxplot. Mostra i boxplot per ogni colonna nella stessa figura.
def plot_box_plots(df, columns):
    fig = make_subplots(rows=1, cols=len(columns), subplot_titles=columns)

    for i, col in enumerate(columns):
        trace = go.Box(x=df[col], name=col)
        fig.add_trace(trace, row=1, col=i+1)

    fig.update_layout(height=300, width=1200, title_text="Boxplots", showlegend=False)
    fig.show()

# 4. `plot_linear_regression_results(X_test, y_test, predictions)`: Questa funzione crea un grafico a dispersione utilizzando Plotly Graph Objects per visualizzare i risultati della regressione lineare. Prende in input i dati di test sull'asse x (`X_test`), i valori osservati di test sull'asse y (`y_test`) e le predizioni della regressione lineare (`predictions`). Mostra il grafico interattivo.
def plot_linear_regression_results(X_test, y_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs Actual'))
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Line'))
    fig.update_layout(title='Linear Regression: Predicted vs Actual', xaxis_title='Actual Values', yaxis_title='Predicted Values')
    fig.show()

# 5. `plot_neural_network_results(X_test, y_test, predictions)`: Questa funzione crea un grafico a dispersione utilizzando Plotly Graph Objects per visualizzare i risultati della regressione neurale. Prende in input i dati di test sull'asse x (`X_test`), i valori osservati di test sull'asse y (`y_test`) e le predizioni della regressione neurale (`predictions`). Mostra il grafico interattivo.
def plot_neural_network_results(X_test, y_test, predictions):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='Predicted vs Actual'))
    fig.add_trace(go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal Line'))
    fig.update_layout(title='Neural Network Regression: Predicted vs Actual', xaxis_title='Actual Values', yaxis_title='Predicted Values')
    fig.show()
