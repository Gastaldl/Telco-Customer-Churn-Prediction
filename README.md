# Telco Customer Churn

Projeto de estudo de ciencia de dados para analisar, modelar e visualizar risco de churn em clientes de telecomunicacoes usando Python, scikit-learn, TensorFlow/Keras e notebooks Jupyter.

O objetivo final nao e apenas prever uma classe binaria de churn. A proposta principal e estimar a probabilidade de cancelamento de cada cliente e transformar essa probabilidade em faixas de acao:

- `Low`: sem acao imediata
- `Medium`: acao leve ou monitoramento
- `High`: acao direta de retencao

Essa abordagem e mais util para uma tomada de decisao de negocio, porque permite priorizar clientes por nivel de risco em vez de tratar todos como apenas "churn" ou "nao churn".

## Dataset

O projeto usa o dataset **Telco Customer Churn**, originalmente disponibilizado no Kaggle.

Arquivo principal:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

A base possui 7.043 clientes e 21 colunas. A variavel-alvo original e `Churn`, com valores `Yes` e `No`.

Principais grupos de variaveis:

- identificador: `customerID`
- numericas: `tenure`, `MonthlyCharges`, `TotalCharges`
- categoricas: `Contract`, `InternetService`, `PaymentMethod`, entre outras
- alvo: `Churn`

Um ponto importante do dataset e que `TotalCharges` vem como texto e possui alguns valores em branco. Por isso, essa coluna precisa ser convertida para numerica antes da modelagem.

## Estrutura do projeto

```text
Telco Customer Churn/
+-- data/
|   +-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|   +-- df_eda.csv
|   +-- churn_predictions.csv
+-- docs/
|   +-- guia-churn.md
+-- models/
|   +-- churn_nn.keras
|   +-- preprocessor.joblib
+-- notebooks/
|   +-- 01_eda.ipynb
|   +-- 02_model.ipynb
|   +-- 03_visualizations.ipynb
+-- src/
|   +-- __init__.py
|   +-- preprocess.py
|   +-- predict.py
+-- README.md
```

## Fluxo do projeto

O projeto esta organizado em tres etapas principais.

### 1. Analise exploratoria

Notebook:

```text
notebooks/01_eda.ipynb
```

Objetivo:

- carregar a base original;
- entender estrutura, tipos de dados e valores iniciais;
- tratar `TotalCharges`;
- analisar a distribuicao da variavel `Churn`;
- investigar churn por contrato, tempo de permanencia e servico de internet;
- gerar uma base enriquecida em `data/df_eda.csv`.

Principais aprendizados dessa etapa:

- a base e desbalanceada;
- clientes com contrato mensal tendem a cancelar mais;
- clientes com menor tempo de permanencia apresentam maior risco;
- `TotalCharges` precisa de tratamento antes de entrar no modelo.

### 2. Modelagem

Notebook:

```text
notebooks/02_model.ipynb
```

Objetivo:

- carregar os dados limpos;
- separar features, alvo e identificadores;
- dividir os dados em treino e teste;
- aplicar pre-processamento com `ColumnTransformer`;
- treinar uma rede neural MLP com TensorFlow/Keras;
- avaliar AUC, precision, recall e matriz de confusao;
- estudar diferentes thresholds;
- salvar o modelo e o pre-processador.

Modelo usado:

```text
Rede neural MLP para classificacao binaria
```

Arquivos gerados:

```text
models/churn_nn.keras
models/preprocessor.joblib
```

Resultado interpretado no notebook:

- o modelo gera uma probabilidade de churn;
- thresholds menores aumentam recall, mas geram mais falsos positivos;
- thresholds maiores aumentam precision, mas deixam passar mais churns;
- uma estrategia por faixas de risco e mais adequada para o objetivo do projeto.

### 3. Visualizacoes

Notebook:

```text
notebooks/03_visualizations.ipynb
```

Objetivo:

- carregar `data/churn_predictions.csv`;
- visualizar a distribuicao das faixas de risco;
- analisar probabilidade media por tempo de permanencia;
- comparar risco por tipo de contrato;
- estimar receita mensal em risco;
- listar clientes prioritarios para acao de retencao.

Esse notebook substitui a ideia inicial de criar um dashboard no Power BI. As visualizacoes passam a ser feitas diretamente em Python.

## Codigo reutilizavel

### `src/preprocess.py`

Contem a logica de preparacao dos dados.

Responsabilidades:

- definir colunas numericas, categoricas e binarias;
- carregar o CSV;
- converter `TotalCharges` para numerico;
- transformar `Churn` em alvo numerico:
  - `Yes` vira `1`;
  - `No` vira `0`;
- construir o pre-processador usado no treino e na predicao.

O pre-processador aplica:

- `StandardScaler` em variaveis numericas;
- `OneHotEncoder` em variaveis categoricas;
- passagem direta para `SeniorCitizen`.

### `src/predict.py`

Usa o modelo treinado para gerar o arquivo final de predicoes.

Entrada:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
models/churn_nn.keras
models/preprocessor.joblib
```

Saida:

```text
data/churn_predictions.csv
```

Colunas principais geradas:

- `customerID`
- `churn_real`
- `churn_prob`
- `risk_band`
- `recommended_action`
- `tenure_group`
- `monthly_charge_band`
- `Contract`
- `tenure`
- `MonthlyCharges`
- `InternetService`
- `PaymentMethod`

Regras de faixa de risco:

```text
churn_prob < 0.50          -> Low
0.50 <= churn_prob < 0.68 -> Medium
churn_prob >= 0.68        -> High
```

## Como executar

### 1. Ativar o ambiente

Exemplo usando conda:

```powershell
conda activate NovelInsight
```

O ambiente precisa conter, no minimo:

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- joblib
- jupyter

### 2. Rodar os notebooks

Execute nesta ordem:

```text
notebooks/01_eda.ipynb
notebooks/02_model.ipynb
notebooks/03_visualizations.ipynb
```

O notebook `02_model.ipynb` deve ser executado antes do `src/predict.py`, porque ele gera o modelo e o pre-processador salvos.

### 3. Gerar o CSV final

Na raiz do projeto, execute:

```powershell
python -m src.predict
```

Ou, se estiver usando o caminho completo do Python do ambiente:

```powershell
C:/Users/marci/miniconda3/envs/NovelInsight/python.exe src/predict.py
```

Ao final, o script deve gerar:

```text
data/churn_predictions.csv
```

com 7.043 linhas.

## Interpretacao do resultado

O campo mais importante do arquivo final e `churn_prob`.

Exemplo:

```text
0.12 -> baixo risco
0.56 -> risco medio
0.82 -> alto risco
```

A coluna `risk_band` transforma essa probabilidade em uma faixa operacional. A coluna `recommended_action` traduz a faixa em uma acao sugerida.

Essa estrutura ajuda a responder perguntas como:

- quantos clientes estao em alto risco?
- quais contratos concentram mais clientes em alto risco?
- clientes novos tem maior probabilidade media de churn?
- qual receita mensal esta associada aos clientes de alto risco?
- quais clientes devem ser priorizados em uma acao de retencao?

## Decisoes importantes do projeto

### Por que nao usar apenas `churn_pred`?

Uma previsao binaria perde informacao.

Dois clientes com probabilidades `0.51` e `0.91` poderiam ser classificados como churn, mas o segundo e claramente mais prioritario. Por isso, o projeto usa probabilidade e faixas de acao.

### Por que salvar o pre-processador?

O modelo foi treinado com dados escalados e codificados por one-hot encoding. Para prever novos dados corretamente, e necessario aplicar exatamente o mesmo pre-processamento usado no treino.

Por isso, o projeto salva:

```text
models/preprocessor.joblib
```

### Por que usar `class_weight` no treino?

A classe churn e menor que a classe nao churn. Sem compensacao, o modelo poderia favorecer demais a classe majoritaria. O uso de pesos de classe ajuda o modelo a prestar mais atencao nos exemplos de cancelamento.

## Possiveis melhorias

- comparar a rede neural com regressao logistica;
- testar outros thresholds;
- calibrar as probabilidades;
- adicionar SHAP para explicabilidade;
- criar mais visualizacoes no `03_visualizations.ipynb`;
- transformar as visualizacoes em um relatorio HTML;
- criar um pipeline unico para treino e predicao.

## Estado atual

O projeto atualmente possui:

- analise exploratoria documentada;
- modelo neural treinado;
- pre-processador salvo;
- script de predicao funcionando;
- CSV final com probabilidade de churn e faixas de acao;
- notebook de visualizacoes em Python.
