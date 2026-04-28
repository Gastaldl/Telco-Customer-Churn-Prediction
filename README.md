# Telco Customer Churn

Projeto de estudo de ciencia de dados para analisar risco de churn em clientes de telecomunicacoes usando Python, scikit-learn, TensorFlow/Keras e notebooks Jupyter.

O foco do projeto e aprender o fluxo completo de trabalho:

1. entender os dados;
2. preparar as variaveis;
3. treinar um modelo de rede neural;
4. avaliar os resultados;
5. gerar probabilidades de churn;
6. visualizar os resultados em Python.

Em vez de tratar o problema apenas como uma previsao binaria, o projeto trabalha principalmente com a probabilidade de churn (`churn_prob`) e com faixas de acao (`risk_band`).

## Dataset

O dataset usado e o **Telco Customer Churn**, com 7.043 clientes e 21 colunas.

Arquivo original:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

A variavel-alvo original e `Churn`, com valores `Yes` e `No`.

Algumas colunas importantes:

- `customerID`: identificador do cliente
- `tenure`: tempo de permanencia
- `MonthlyCharges`: cobranca mensal
- `TotalCharges`: cobranca total acumulada
- `Contract`: tipo de contrato
- `InternetService`: tipo de servico de internet
- `PaymentMethod`: metodo de pagamento
- `Churn`: indica se o cliente cancelou

Um detalhe importante e que `TotalCharges` vem como texto no CSV original. Por isso, essa coluna e convertida para numero no pre-processamento.

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
+-- README.md
```

## Notebooks

### `notebooks/01_eda.ipynb`

Notebook de analise exploratoria.

Ele e usado para:

- carregar o dataset original;
- entender tipos de dados e estrutura da base;
- tratar `TotalCharges`;
- analisar a distribuicao de `Churn`;
- observar churn por contrato, tempo de permanencia e servico de internet;
- gerar `data/df_eda.csv`.

### `notebooks/02_model.ipynb`

Notebook de modelagem.

Ele e usado para:

- carregar os dados tratados;
- separar features, alvo e identificadores;
- dividir treino e teste;
- aplicar pre-processamento;
- treinar uma rede neural MLP com Keras;
- avaliar AUC, precision, recall e matriz de confusao;
- estudar thresholds;
- salvar o modelo e o pre-processador.

Arquivos gerados:

```text
models/churn_nn.keras
models/preprocessor.joblib
```

### `notebooks/03_visualizations.ipynb`

Notebook de visualizacoes.

Ele usa o CSV final com as probabilidades do modelo para analisar:

- distribuicao das faixas de risco;
- probabilidade media por tempo de permanencia;
- risco por tipo de contrato;
- receita mensal associada a clientes de alto risco;
- clientes prioritarios para acao de retencao.

## Codigo em `src`

### `src/preprocess.py`

Centraliza a preparacao dos dados.

Principais responsabilidades:

- definir colunas numericas, categoricas e binarias;
- carregar a base original;
- converter `TotalCharges` para numero;
- transformar `Churn` em alvo numerico:
  - `Yes` vira `1`;
  - `No` vira `0`;
- criar o pre-processador com:
  - `StandardScaler` para colunas numericas;
  - `OneHotEncoder` para colunas categoricas;
  - passagem direta para `SeniorCitizen`.

### `src/predict.py`

Script opcional para gerar um CSV com os resultados do modelo treinado.

Ele nao treina o modelo. Ele carrega:

```text
models/churn_nn.keras
models/preprocessor.joblib
```

Depois aplica o modelo na base original e gera:

```text
data/churn_predictions.csv
```

Esse CSV e util para analisar os resultados fora do notebook de modelagem ou para alimentar o notebook de visualizacoes.

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

As faixas de risco sao criadas a partir de `churn_prob`:

```text
churn_prob < 0.50          -> Low
0.50 <= churn_prob < 0.68 -> Medium
churn_prob >= 0.68        -> High
```

As acoes recomendadas sao:

```text
Low    -> No immediate action
Medium -> Light action / monitoring
High   -> Direct retention action
```

## Como executar

Ative o ambiente Python usado no projeto. Exemplo com conda:

```powershell
conda activate NovelInsight
```

Dependencias principais:

- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib
- seaborn
- joblib
- jupyter

Execute os notebooks nesta ordem:

```text
notebooks/01_eda.ipynb
notebooks/02_model.ipynb
notebooks/03_visualizations.ipynb
```

Se quiser regenerar o CSV com os resultados do modelo, rode na raiz do projeto:

```powershell
python -m src.predict
```

Ou execute diretamente com o Python do ambiente:

```powershell
C:/Users/marci/miniconda3/envs/NovelInsight/python.exe src/predict.py
```

## Interpretacao dos resultados

A coluna mais importante do CSV final e:

```text
churn_prob
```

Ela representa a probabilidade estimada de churn para cada cliente.

Exemplo:

```text
0.12 -> baixo risco
0.56 -> risco medio
0.82 -> alto risco
```

O projeto usa `risk_band` e `recommended_action` porque uma probabilidade e mais informativa do que uma classificacao simples em `0` ou `1`.

Por exemplo, dois clientes com `churn_prob` de `0.51` e `0.91` poderiam ser classificados como risco, mas o cliente com `0.91` deve ser priorizado.

## Estado atual

O projeto possui:

- analise exploratoria em notebook;
- modelo neural treinado;
- pre-processador salvo;
- script opcional para gerar CSV com probabilidades;
- CSV final com faixas de risco e acoes recomendadas;
- notebook de visualizacoes em Python.

## Possiveis proximos passos

- melhorar as visualizacoes no `03_visualizations.ipynb`;
- comparar a rede neural com um modelo baseline, como regressao logistica;
- testar outros thresholds para as faixas de risco;
- calibrar as probabilidades do modelo;
- adicionar explicabilidade com SHAP;
- criar um relatorio final com os principais insights.
