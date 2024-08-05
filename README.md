# Prevendo Preço de Passsagens Aéreas com Machine Learning – Flight Price Prediction ✈️

O dataset [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) disponibilizado por [Shubham Bathwal](https://www.kaggle.com/shubhambathwal) contém dados de reservas aéreas obtidas do website "Easy My Trip". Os dados cobrem o período de 11 de Fevereiro até 31 de Março de 2022, com 300261 registros.

![flight](https://i.imgur.com/RkAHy6w.jpeg)

## 1.1. Metas e objetivos

Este projeto tem objetivo de responder algumas perguntas de negócio e criar um modelo de Machine Learning para predição de preços de voos.

### Perguntas de negócio
1. Preço varia de acordo com a Linha Aérea? e com a Classe?
2. Como os preços das passagens são afetados, entre 1 e 2 dias antes da viagem?
3. O preço muda de acordo com o período do dia para chegada e partida?
4. O preço muda de acordo com o destino de partida e chegada?

### Resultados
Através de uma breve análise exploratória de dados foram respondidas as perguntas de negócio, e através da modelagem cheguei a um modelo com as seguintes métricas:

|Métrica|Resultado|
|--|---|
|Mean Absolute Error|1588.4280|
|Mean Squared Error|8835481.4604|
|Root Mean Squared Error|2972.4537|
|R2 Score|0.9828|

### Ferramentas utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

### Features
|Coluna|Descrição|
|-------|--------|
|airline|A linha aérea do voo|
|flight|O código de identificação do voo|
|source_city|A cidade de onde o voo está partindo|
|departure_time|Período do dia em qual o voo partiu|
|stops|Número de paradas entre a partida e o destino|
|arrival_time|Período do dia em que o voo chegou|
|destination_city|A cidade destino do voo|
|class|Classe do voo|
|duration|Duração em horas do voo|
|days_left|Diferença entre o dia da viagem e da reserva|
|price|Preço da passagem|

### Bibliotecas Python utilizadas
#### Manipulação de dados
- Pandas, Numpy.
#### EDA
- Seaborn, Matplotlib.
#### Machine Learning
- XGBoost, sklearn, feature_engine.

# Exploratory Data Analysis
## Comportamento da variável alvo

![hist](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot1.png?raw=true)

![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot2.png?raw=true)

## Target, features e as perguntas de negócio
#### 1. Preço varia de acordo com a Linha Aérea? e com a Classe?
![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot3.png?raw=true)

![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot4.png?raw=true)

Vistara e Air India são as únicas empreas que oferecem o voos de classe Executiva e por isso tem os maiores valores de passagem aérea. 

#### 2. Como os preços das passagens são afetados entre 1 e 2 dias antes da viagem?

![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot5.png?raw=true)

O preço de passagem aéreas tem tendência de serem maiores quanto mais próximo do voo.

#### 3. O preço muda de acordo com o período do dia para chegada e partida?
![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot6.png?raw=true)

A madrugada é o período onde se encontra as passagens mais baratas, já os voos a noite são os mais caros tanto para chegada quanto para partida. 

#### 4. O preço muda de acordo com o destino de partida e chegada?
![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot7.png?raw=true)

Sim, os pontos de partida e destino tem influência no preço. Delhi é o destino mais barato, seguido por Hyderabad.

## Matriz de correlação
![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot8.png?raw=true)

# Modelo de Machine Learning
Escolhi o XGBoost para este projeto, e as métricas resultantes após modelagem e tunagem de hiper parâmetros foram: 

|Métrica|Resultado|
|--|---|
|Mean Absolute Error|1588.4280|
|Mean Squared Error|8835481.4604|
|Root Mean Squared Error|2972.4537|
|R2 Score|0.9828|

![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot11.png?raw=true)
![box](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot12.png?raw=true)
