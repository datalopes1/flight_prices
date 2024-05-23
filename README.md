
# Prevendo Preços de Voos Aéreos com Regressão Linear ✈️

Neste processo serão realizados os processos de Análise Exploratória de Dados e construção de um modelo preditivo de Machine Learning com o XGBoost a partir do dataset Flight Price Prediction. Os dados podem ser encontrados no [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) e foram disponibilizados por [Shubham Bathwal](https://www.kaggle.com/shubhambathwal).

![](https://images.unsplash.com/photo-1483450388369-9ed95738483c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

### Objetivos e resultados
O primeiro objetivo é responder as seguintes perguntas sobre o dataset:

- Preço varia de acordo com a Linha Aérea? e com a Classe?
- Como os preços das passagens são afetados, entre 1 e 2 dias antes da viagem?
- O preço muda de acordo com o período do dia para chegada e partida?
- O preço muda de acordo com o destino de partida e chegada?

O segundo foi a construção de um modelo de Regressão Linear utilizando XGBRegressor no qual consegui um coeficiente de determinação (R²) de 0,977. 

### 🛠️ Ferramentas utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

## A estrutura do dataset

A colunas do dataset estão organizadas da seguinte forma:

|Coluna|Descrição|
|-------|---------|
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

## Bibliotecas Python utilizadas
#### Manipulação de dados
- Pandas, Numpy.
#### EDA
- Seaborn, Matplotlib.
#### Machine Learning
- XGBoost, sklearn, feature_engine.

# Análise Exploratória de Dados (EDA)
## Preço varia de acordo com a Linha Aérea? e com a Classe?
![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot9.png?raw=true)

Vistara e Air India tem preços de passagens mais caras que o restante das linhas aéreas. O restante tem preço parecido mas existe a linha área define bastante o preço. Vamos olhar sob o prisma da classe também.

![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot10.png?raw=true)

## Como os preços das passagens são afetados, entre 1 e 2 dias antes da viagem?
![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot11.png?raw=true)

Passagens quando são compras com maior antecedência são mais baratas.
## O preço muda de acordo com o período do dia para chegada e partida?
![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot12.png?raw=true)

A madrugada também é o melhor período para comprar passagens de chegada. Se tornando o período ideal para comprar passagens seguido pela tarde e o começo da manhã.

## O preço muda de acordo com o destino de partida e chegada?
![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot13.png?raw=true)

Sim, os pontos de partida e destino tem influência no preço. Delhi é o destino mais barato, seguido por Hyderabad.

![](https://images.unsplash.com/photo-1504150558240-0b4fd8946624?q=80&w=1964&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

# Modelo de Previsão
## Pré-processamento dos dados
Selecionei as colunas que seriam utilizadas, e apliquei o OneHotEncoder. 
## Usando o XGBRegressor
Os dados foram separados em conjuntos de treino e testes e o modelo foi instanciado e ajustado.
## Métricas
As métricas do modelo foram as seguintes

|Métrica|Resultado|
|------------------|-------------|
|Mean Squared Error|11433028.9639|
|R² Score|0.9778|

![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot14.png?raw=true)

![](https://github.com/datalopes1/flight_prices/blob/datalopes1/doc/img/plot15.png?raw=true)

# Conclusões
### Respondendo as perguntas iniciais

- As companhias aéreas com maior valor de passagem são a Vistara e Air India, o que é natural já que são as únicas que oferecem voos de classe executiva, sendo esses os voos com passagens mais caras.
- Comprar passagens com antecedência vai trazer melhores ofertas nos preços,quanto mais próxima ao voo mais cara é a passagem.
- Viajar de madrugada e cedo na manhã é mais barato que em outros períodos do dia.
- Delhi e Hyderabad são os destinos de viagem mais baratos no conjunto de dados, e Chennai o destino mais caro.

### Sobre o modelo

Com a utilização do algoritmo XGBRegressor conseguimos uma ótima métrica de coeficiente de determinação, em 0.9778, com a aplicação deste modelo será possível prever com segurança os preços de passagens aéreas. 
