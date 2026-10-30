# Stroke Prediction — Projeto de Machine Learning

Documentação do projeto da disciplina de Machine Learning (Insper) sobre o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset): 5.110 pacientes com dados clínicos, demográficos e de hábitos, e a indicação de quais tiveram AVC.

**Objetivo:** construir um pipeline completo de Machine Learning para prever se um paciente teve AVC (classificação binária) a partir de idade, condições de saúde, tabagismo e dados socioeconômicos.

- **Aluno:** Matheus Henrique Pereira Borba
- **Código:** [github.com/mthperera/APS-ML](https://github.com/mthperera/APS-ML)

| Etapa | Entrega | Conteúdo | Peso |
|---|---|---|---|
| APS1 | 14/09/2026 | análise exploratória (tarefas 1 a 3) e pré-processamento (tarefa 4) | 30% |
| APS2 | 30/10/2026 | modelagem, avaliação no teste, interpretação e discussão crítica | 70% |

## Resultado em uma frase

Uma **Random Forest** sobre o pipeline da APS1, sem três variáveis que a análise mostrou não terem sinal próprio, com limiar escolhido para triagem: no teste, encontra **33 dos 50 AVCs (recall de 0,66)** sinalizando 19% dos pacientes, com **ROC-AUC de 0,835** e PR-AUC de 0,214 (Dummy: 0,049). A regressão logística empata com ela no teste, e a principal limitação é a dependência de um indicador de IMC ausente cujo sinal não se repete no teste.

## Como navegar

| Página | O que tem |
|---|---|
| [Dataset](dataset.md) | origem dos dados, o que é cada coluna e os problemas de qualidade encontrados |
| [Análise exploratória](eda.md) | os achados das tarefas 1 a 3, com gráficos, tabelas e testes estatísticos |
| [Pré-processamento](preprocessing.md) | cada decisão da tarefa 4 e a justificativa baseada na análise |
| [Modelagem](modeling.md) | APS2, seção 1: métrica, os três modelos, busca de hiperparâmetros, mudança no pipeline, curva de aprendizado e limiar |
| [Avaliação no teste](evaluation.md) | APS2, seção 2: métricas com intervalos de confiança, matriz de confusão e curva ROC |
| [Interpretação](interpretation.md) | APS2, seção 3: coeficientes, importâncias e SHAP |
| [Discussão crítica](discussion.md) | APS2, seção 4: limitações do modelo e dos dados, operacionalização e referências |

## Onde está cada item do enunciado da APS2

| Item do enunciado | Onde | Resultado |
|---|---|---|
| **1. Seleção e implementação dos modelos** | | |
| usar o mesmo pipeline da APS1 | [Modelagem › Ponto de partida](modeling.md#ponto-de-partida-o-pipeline-da-aps1) | o `ColumnTransformer` da APS1 vira o 1º passo de um `Pipeline` com cada modelo |
| 3 modelos: Dummy, linear com regularização, ensemble de árvores | [Modelagem › 2](modeling.md#2-os-tres-modelos) | `DummyClassifier`, `LogisticRegression` (L1/L2/elastic net), `RandomForestClassifier` |
| uma métrica, reportada para os 3 modelos | [Modelagem › 1](modeling.md#1-metrica-e-estrategia-de-validacao) e [4](modeling.md#4-comparacao-dos-modelos) | PR-AUC: Dummy 0,049, logística 0,225, Random Forest 0,258 |
| seleção de hiperparâmetros no linear e no ensemble | [Modelagem › 3](modeling.md#3-selecao-de-hiperparametros) | `GridSearchCV` (66 combinações) e `RandomizedSearchCV` (40 de 360) |
| propor e implementar uma mudança no pipeline | [Modelagem › 5](modeling.md#5-mudanca-no-pipeline) | remover gender, Residence_type e work_type, com ablação |
| avaliar a mudança com nova seleção de hiperparâmetros | [Modelagem › 5](modeling.md#5-mudanca-no-pipeline) | 0,258 → 0,269, empate técnico (teste t corrigido, p = 0,18); adotada por parcimônia |
| curvas de aprendizado do melhor modelo | [Modelagem › 6](modeling.md#6-curva-de-aprendizado) | variância e viés; mais dados ajudam pouco |
| **2. Avaliação no teste** | | |
| Accuracy, Precision, Recall e F1 | [Avaliação › 1](evaluation.md#1-metricas) | 0,830 · 0,174 · 0,660 · 0,275, com intervalos de confiança |
| matriz de confusão, com falsos positivos e falsos negativos | [Avaliação › 2](evaluation.md#2-matriz-de-confusao) | 157 falsos positivos e 17 falsos negativos, com o perfil de cada grupo |
| curva ROC e ROC-AUC | [Avaliação › 3](evaluation.md#3-curva-roc) | ROC-AUC 0,835 |
| **3. Interpretação** | | |
| linear: 3 coeficientes de maior magnitude | [Interpretação › 1](interpretation.md#1-regressao-logistica-os-3-maiores-coeficientes) | idade, IMC ausente, glicose (na escala comum) |
| ensemble: 3 features de maior importância | [Interpretação › 2](interpretation.md#2-random-forest-as-3-features-mais-importantes) | idade, glicose, IMC ausente, conferidas por permutação |
| SHAP: importância global | [Interpretação › 3](interpretation.md#importancia-global) | beeswarm e média do \|SHAP\| |
| SHAP: 3 predições individuais | [Interpretação › 3](interpretation.md#tres-predicoes-individuais) | verdadeiro positivo, falso negativo e falso positivo |
| **4. Discussão crítica** | | |
| 2 limitações do modelo | [Discussão › Modelo](discussion.md#limitacoes-do-modelo) | dependência do IMC ausente; ponto de operação preso à prevalência |
| 2 limitações dos dados | [Discussão › Dados](discussion.md#limitacoes-dos-dados) | dados transversais; fatores de risco ausentes |
| 2 ações de operacionalização e manutenção | [Discussão › Operacionalização](discussion.md#operacionalizacao-e-manutencao) | monitoramento com re-treino; versionamento com model card |
| referências bibliográficas | [Discussão › Referências](discussion.md#referencias) | 22 referências citadas no texto |
