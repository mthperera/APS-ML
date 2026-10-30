# Stroke Prediction — Projeto de Machine Learning

Projeto da disciplina de Machine Learning (Insper) com o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset): 5.110 pacientes com dados clínicos, demográficos e de hábitos, e a indicação de quais tiveram AVC. O objetivo é construir um pipeline completo de Machine Learning para prever se um paciente teve AVC (classificação binária).

**Resultado:** uma Random Forest sobre o pipeline da APS1 (sem gênero, residência e tipo de trabalho), com limiar escolhido para triagem, encontra 33 dos 50 AVCs do teste (recall 0,66), com ROC-AUC 0,835 e PR-AUC 0,214 (Dummy: 0,049).

**Documentação completa:** https://mthperera.github.io/APS-ML/

- **Aluno:** Matheus Henrique Pereira Borba
- **Data da entrega (APS1):** 14/09/2026
- **Data da entrega (APS2):** 27/10/2026

## Estrutura do repositório

```text
APS-ML/
├── .github/workflows/docs.yml   publica a documentação no GitHub Pages a cada push na main
├── data/
│   ├── raw/                     CSV original baixado do Kaggle
│   └── processed/               arquivos gerados pelos notebooks
├── docs/                        páginas e imagens do site de documentação (MkDocs)
├── models/                      modelos ajustados, gerados pelo notebook 03
├── notebooks/
│   ├── 01_eda.ipynb             APS1: análise exploratória (tarefas 1 a 3)
│   ├── 02_preprocessing.ipynb   APS1: pré-processamento (tarefa 4)
│   ├── 03_modeling.ipynb        APS2: modelagem (seção 1)
│   └── 04_evaluation.ipynb      APS2: avaliação, interpretação e discussão (seções 2 a 4)
├── scripts/
│   └── download_data.py         baixa o dataset do Kaggle para data/raw/
├── mkdocs.yml                   configuração do site de documentação
├── requirements.txt             bibliotecas dos notebooks
└── requirements-docs.txt        bibliotecas do site de documentação
```

### O que é cada coisa

**data/raw/** recebe o arquivo `healthcare-dataset-stroke-data.csv`, baixado do Kaggle. O CSV não é versionado porque o autor do dataset indica fonte confidencial e uso apenas educacional; ele é baixado na primeira execução.

**data/processed/** recebe os arquivos gerados pelos notebooks, também fora do git:

| Arquivo | Gerado por | Conteúdo |
|---|---|---|
| train.csv | 01_eda | 80% dos pacientes (4.087), split estratificado pelo target |
| test.csv | 01_eda | 20% dos pacientes (1.022), reservados para a avaliação final |
| train_processed.csv | 02_preprocessing | treino após o pipeline: 18 features + stroke |
| test_processed.csv | 02_preprocessing | teste após o pipeline: 18 features + stroke |

**notebooks/01_eda.ipynb** carrega o dataset, descreve as features, verifica tipos, valores ausentes, inconsistências e desbalanceamento, faz o split treino/teste e a análise univariada, bivariada e multivariada usando só o treino.

**notebooks/02_preprocessing.ipynb** lê o split salvo pelo 01 e aplica as decisões da análise: imputação do IMC com indicador de ausência, log na glicose, clipping no IMC, one-hot encoding, padronização, PCA e o pipeline final com ColumnTransformer do scikit-learn.

**notebooks/03_modeling.ipynb** reaproveita o pipeline do 02 dentro de um `Pipeline` com cada modelo e cobre a seção 1 da APS2: escolha da métrica (PR-AUC) e da validação cruzada, Dummy, regressão logística com regularização (GridSearchCV) e Random Forest (RandomizedSearchCV), a mudança no pipeline com nova busca de hiperparâmetros, a curva de aprendizado e a escolha do limiar de decisão. Usa só o treino e salva os modelos em `models/modelos.joblib`.

**notebooks/04_evaluation.ipynb** carrega os modelos e usa o teste pela primeira vez: métricas, matriz de confusão e curva ROC (seção 2); coeficientes da regressão logística, importâncias da Random Forest e SHAP (seção 3); e os números de apoio da discussão crítica, com as referências (seção 4).

**models/** recebe `modelos.joblib`, gerado pelo notebook 03 e lido pelo 04. Não é versionado, como os dados.

**scripts/download_data.py** baixa o CSV para `data/raw/` usando a biblioteca kagglehub. Não precisa de conta no Kaggle. Se o arquivo já existir, não baixa de novo.

**docs/** e **mkdocs.yml** formam o site de documentação, com oito páginas: início, dataset, análise exploratória, pré-processamento, modelagem, avaliação no teste, interpretação e discussão crítica. As imagens em `docs/img/` são as figuras dos notebooks.

**.github/workflows/docs.yml** é o workflow que faz o build do site e o publica no GitHub Pages sempre que há um push na branch `main`.

**requirements.txt** lista as bibliotecas dos notebooks (pandas, numpy, matplotlib, seaborn, scikit-learn, shap, joblib, kagglehub, ipykernel). **requirements-docs.txt** lista as do site (mkdocs, mkdocs-material).

## Como rodar

### 1. Preparar o ambiente

Requer Python 3.12 ou mais novo.

```powershell
git clone https://github.com/mthperera/APS-ML.git
cd APS-ML
python -m venv env
env\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux ou Mac, troque a ativação por `source env/bin/activate`.

### 2. Baixar o dataset

```powershell
python scripts/download_data.py
```

Esse passo é opcional: se `data/raw/` estiver vazia, o notebook 01 faz o download sozinho. Por isso os notebooks também rodam no Google Colab sem o restante do repositório.

### 3. Executar os notebooks

Abra os notebooks no VS Code ou no Jupyter, selecione o kernel do ambiente `env` e execute na ordem:

1. `notebooks/01_eda.ipynb`, que gera `data/processed/train.csv` e `test.csv`;
2. `notebooks/02_preprocessing.ipynb`, que lê esses arquivos e gera `train_processed.csv` e `test_processed.csv`;
3. `notebooks/03_modeling.ipynb`, que ajusta e compara os modelos e salva `models/modelos.joblib` (cerca de 4 minutos);
4. `notebooks/04_evaluation.ipynb`, que avalia o modelo final no teste e o interpreta.

Todos são determinísticos (random_state = 42), então os resultados se repetem a cada execução.

### 4. Ver a documentação localmente

```powershell
pip install -r requirements-docs.txt
mkdocs serve
```

O site fica em http://127.0.0.1:8000 e recarrega sozinho a cada alteração em `docs/` ou no `mkdocs.yml`.

### 5. Publicar a documentação

A publicação é automática: todo push na `main` roda o workflow, que gera o site e o envia para a branch `gh-pages`. É preciso configurar o Pages uma única vez no GitHub: Settings → Pages → Source: Deploy from a branch → branch `gh-pages`, pasta `/ (root)`. O site fica em https://mthperera.github.io/APS-ML/.
