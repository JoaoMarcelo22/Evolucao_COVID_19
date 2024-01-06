import pandas as pd 
import numpy as np 
##from datetime import detetime
import plotly.express as px 
import plotly.graph_objects as go 
import re
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from prophet import Prophet

url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'

df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])


# Colunas não devem ter letras maiúsculas e nem caracteres especiais.

def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "",col_name).lower()

df.columns = [corrige_colunas(col) for col in df.columns]
#print(df)

# Apenas dados do Brasil

df.countryregion.unique() # Mostra todos os paises.

#print(df.loc[df.countryregion == 'Brazil']) Mostra todos os dados do Brasil

brasil = df.loc[
    (df.countryregion == 'Brazil') &
    (df.confirmed > 0)
]
#print(brasil)

fig = px.line(brasil, 'observationdate','confirmed', title=' Casos Confirmados no Brasil')
fig.show()

brasil['novoscasos'] = list(map(
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],   #Lambda e uma funcao anonima, Funcao para entender os casos por dia
    np.arange(brasil.shape[0])
))
#print(brasil)
fig2 = px.line(brasil, x='observationdate',y='novoscasos', title='Novos casos por dia')
fig2.show()

fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name="Mortes",
                mode='lines+markers', line={'color':'black'})
)
fig3.update_layout(title="Mortes por Covid-19 no Brasil")
fig3.show()

def taxa_crescimento(data,variable, data_inicio=None, data_fim=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)

    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim,variable].values[0]

    n = (data_fim - data_inicio).days # pontos no tempo que vamos avaliar

    taxa = (presente/passado)**(1/n) - 1  # calcular a taxa

    return taxa*100

#print(taxa_crescimento(brasil,'confirmed'))

def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    data_fim = data.observationdate.max()

    n = (data_fim - data_inicio).days

    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        range(1, n+1)
    ))
    return np.array(taxas) * 100

tx_dia = taxa_crescimento_diaria(brasil,'confirmed')

primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()

fig4 = px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
               y=tx_dia, title='Taxa de crescimento de casos confirmados no brasil')

fig4.show()

confirmados = brasil.confirmed 
confirmados.index = brasil.observationdate
#print(confirmados)
res = seasonal_decompose(confirmados)
fig5, (ax1, ax2, ax3, ax4) = plt.subplots(4,1,figsize=(10,8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

modelo = auto_arima(confirmados)

fig6 = go.Figure(go.Scatter(
    x=confirmados.index, 
    y=confirmados, 
    name='Observados'
))
fig6.add_trace(go.Scatter(
    x=confirmados.index, 
    y=modelo.predict_in_sample(), 
    name='Preditos'
))
future_dates = pd.date_range(start='2020-05-20', end='2020-06-20')
future_forecast = modelo.predict(n_periods=31)  # Aqui, usamos 'n_periods' para especificar o número de períodos de previsão

fig6.add_trace(go.Scatter(
    x=future_dates,
    y=future_forecast,
    name='Forecast'
))
fig6.update_layout(title='Previsão de casos confirmados no Brasil para os próximos 30 dias')
fig6.show()

train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

train.rename(columns={'observationdate':'ds','confirmed':'y'}, inplace=True)
test.rename(columns={'observationdate':'ds','confirmed':'y'}, inplace=True)

profeta = Prophet(growth='logistic', changepoints=['2020-03-21','2020-03-30','2020-04-25','2020-05-03','2020-05-10'])

pop = 211463256

train['cap'] = pop

profeta.fit(train)

future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop
forecast = profeta.predict(future_dates)

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predicao'))
fig7.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados Treino'))
fig7.update_layout(title='Predicoes de casos confirmados no Brasil')
fig7.show()