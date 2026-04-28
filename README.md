# Telco Customer Churn

Projeto de estudo de ciência de dados para analisar risco de churn em clientes de telecomunicações usando Python, scikit-learn, TensorFlow/Keras e notebooks Jupyter.

O foco do projeto é aprender o fluxo completo de trabalho:

1. entender os dados;
2. preparar as variáveis;
3. treinar um modelo de rede neural;
4. avaliar os resultados;
5. gerar probabilidades de churn;
6. visualizar os resultados em Python.

Em vez de tratar o problema apenas como uma previsão binária, o projeto trabalha principalmente com a probabilidade de churn (`churn_prob`) e com faixas de ação (`risk_band`).

## Dataset

O dataset usado é o **Telco Customer Churn**, com 7.043 clientes e 21 colunas.

Arquivo original:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

A variável-alvo original é `Churn`, com valores `Yes` e `No`.

Algumas colunas importantes:

- `customerID`: identificador do cliente
- `tenure`: tempo de permanência
- `MonthlyCharges`: cobrança mensal
- `TotalCharges`: cobrança total acumulada
- `Contract`: tipo de contrato
- `InternetService`: tipo de serviço de internet
- `PaymentMethod`: método de pagamento
- `Churn`: indica se o cliente cancelou

Um detalhe importante é que `TotalCharges` vem como texto no CSV original. Por isso, essa coluna é convertida para número no pré-processamento.

## Estrutura do projeto

```text
Telco Customer Churn/
+-- data/
|   +-- WA_Fn-UseC_-Telco-Customer-Churn.csv
|   +-- df_eda.csv
|   +-- churn_predictions.csv
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
+-- requirements.txt
+-- README.md
```

## Notebooks

### `notebooks/01_eda.ipynb`

Notebook de análise exploratória.

Ele é usado para:

- carregar o dataset original;
- entender tipos de dados e estrutura da base;
- tratar `TotalCharges`;
- analisar a distribuição de `Churn`;
- observar churn por contrato, tempo de permanência e serviço de internet;
- gerar `data/df_eda.csv`.

### `notebooks/02_model.ipynb`

Notebook de modelagem.

Ele é usado para:

- carregar os dados tratados;
- separar features, alvo e identificadores;
- dividir treino e teste;
- aplicar pré-processamento;
- treinar uma rede neural MLP com Keras;
- avaliar AUC, precision, recall e matriz de confusão;
- estudar thresholds;
- salvar o modelo e o pré-processador.

Arquivos gerados:

```text
models/churn_nn.keras
models/preprocessor.joblib
```

### `notebooks/03_visualizations.ipynb`

Notebook de visualizações.

Ele usa o CSV final com as probabilidades do modelo para analisar:

- distribuição das faixas de risco;
- probabilidade média por tempo de permanência;
- risco por tipo de contrato;
- receita mensal associada a clientes de alto risco;
- clientes prioritários para ação de retenção.

## Código em `src`

### `src/preprocess.py`

Centraliza a preparação dos dados.

Principais responsabilidades:

- definir colunas numéricas, categóricas e binárias;
- carregar a base original;
- converter `TotalCharges` para número;
- transformar `Churn` em alvo numérico:
  - `Yes` vira `1`;
  - `No` vira `0`;
- criar o pré-processador com:
  - `StandardScaler` para colunas numéricas;
  - `OneHotEncoder` para colunas categóricas;
  - passagem direta para `SeniorCitizen`.

### `src/predict.py`

Script opcional para gerar um CSV com os resultados do modelo treinado.

Ele não treina o modelo. Ele carrega:

```text
models/churn_nn.keras
models/preprocessor.joblib
```

Depois aplica o modelo na base original e gera:

```text
data/churn_predictions.csv
```

Esse CSV é útil para analisar os resultados fora do notebook de modelagem ou para alimentar o notebook de visualizações.

Principais colunas geradas:

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

As faixas de risco são criadas a partir de `churn_prob`:

```text
churn_prob < 0.50          -> Low
0.50 <= churn_prob < 0.68 -> Medium
churn_prob >= 0.68        -> High
```

As ações recomendadas são:

```text
Low    -> No immediate action
Medium -> Light action / monitoring
High   -> Direct retention action
```

## Como executar

### 1. Criar e ativar um ambiente virtual

Na raiz do projeto, crie um ambiente virtual:

```powershell
python -m venv .venv
```

Ative o ambiente no PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Atualize o `pip` e instale as dependências:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Abrir os notebooks

Com o ambiente ativo, inicie o Jupyter:

```powershell
jupyter notebook
```

Execute primeiro:

```text
notebooks/01_eda.ipynb
notebooks/02_model.ipynb
```

O notebook `02_model.ipynb` treina o modelo e salva:

```text
models/churn_nn.keras
models/preprocessor.joblib
```

### 3. Gerar o CSV com os resultados do modelo

Depois de treinar e salvar o modelo, rode na raiz do projeto:

```powershell
python -m src.predict
```

Esse comando gera:

```text
data/churn_predictions.csv
```

Esse arquivo contém as probabilidades calculadas pelo modelo, as faixas de risco e as ações recomendadas.

### 4. Rodar o notebook de visualizações

Depois que `data/churn_predictions.csv` existir, execute:

```text
notebooks/03_visualizations.ipynb
```

Esse notebook usa o CSV gerado pelo `predict.py` para criar as visualizações em Python.

## Interpretação dos resultados

A coluna mais importante do CSV final é:

```text
churn_prob
```

Ela representa a probabilidade estimada de churn para cada cliente.

Exemplo:

```text
0.12 -> baixo risco
0.56 -> risco médio
0.82 -> alto risco
```

O projeto usa `risk_band` e `recommended_action` porque uma probabilidade é mais informativa do que uma classificação simples em `0` ou `1`.

Por exemplo, dois clientes com `churn_prob` de `0.51` e `0.91` poderiam ser classificados como risco, mas o cliente com `0.91` deve ser priorizado.
