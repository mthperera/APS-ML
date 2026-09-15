# Stroke Prediction — Projeto de Machine Learning

Projeto da disciplina de Machine Learning (Insper) com o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset): 5.110 pacientes com dados clínicos, demográficos e de hábitos, e a indicação de quais tiveram AVC. O objetivo é construir um pipeline completo de Machine Learning para prever se um paciente teve AVC (classificação binária).

**Documentação completa:** https://mthperera.github.io/APS-ML/

## Estrutura do repositório

```text
APS-ML/
├── .github/workflows/docs.yml   publica a documentação no GitHub Pages a cada push na main
├── data/
│   ├── raw/                     CSV original baixado do Kaggle
│   └── processed/               arquivos gerados pelos notebooks
├── docs/                        páginas e imagens do site de documentação (MkDocs)
├── notebooks/
│   ├── 01_eda.ipynb             análise exploratória (tarefas 1 a 3)
│   └── 02_preprocessing.ipynb   pré-processamento (tarefa 4)
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

**scripts/download_data.py** baixa o CSV para `data/raw/` usando a biblioteca kagglehub. Não precisa de conta no Kaggle. Se o arquivo já existir, não baixa de novo.

**docs/** e **mkdocs.yml** formam o site de documentação, com quatro páginas: início, dataset, análise exploratória e pré-processamento. As imagens em `docs/img/` são as figuras dos notebooks.

**.github/workflows/docs.yml** é o workflow que faz o build do site e o publica no GitHub Pages sempre que há um push na branch `main`.

**requirements.txt** lista as bibliotecas dos notebooks (pandas, numpy, matplotlib, seaborn, scikit-learn, kagglehub, ipykernel). **requirements-docs.txt** lista as do site (mkdocs, mkdocs-material).

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
2. `notebooks/02_preprocessing.ipynb`, que lê esses arquivos e gera `train_processed.csv` e `test_processed.csv`.

Os dois são determinísticos (random_state = 42), então os resultados se repetem a cada execução.

### 4. Ver a documentação localmente

```powershell
pip install -r requirements-docs.txt
mkdocs serve
```

O site fica em http://127.0.0.1:8000 e recarrega sozinho a cada alteração em `docs/` ou no `mkdocs.yml`.

### 5. Publicar a documentação

A publicação é automática: todo push na `main` roda o workflow, que gera o site e o envia para a branch `gh-pages`. É preciso configurar o Pages uma única vez no GitHub: Settings → Pages → Source: Deploy from a branch → branch `gh-pages`, pasta `/ (root)`. O site fica em https://mthperera.github.io/APS-ML/.
