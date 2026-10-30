# Stroke Prediction — Projeto de Machine Learning

Documentação do projeto da disciplina de Machine Learning (Insper) sobre o [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset): 5.110 pacientes com dados clínicos, demográficos e de hábitos, e a indicação de quais tiveram AVC.

**Objetivo:** construir um pipeline completo de Machine Learning para prever se um paciente teve AVC (classificação binária) a partir de idade, condições de saúde, tabagismo e dados socioeconômicos.

- **Aluno:** Matheus Henrique Pereira Borba
- **Código:** [github.com/mthperera/APS-ML](https://github.com/mthperera/APS-ML)

| Etapa | Entrega | Conteúdo | Peso |
|---|---|---|---|
| APS1 | 14/09/2026 | análise exploratória (tarefas 1 a 3) e pré-processamento (tarefa 4) | 30% |
| APS2 | 27/10/2026 | modelagem, avaliação no teste, interpretação e discussão crítica | 70% |

## Resultado em uma frase

Uma **Random Forest** sobre o pipeline da APS1, sem as três variáveis que a análise mostrou não terem sinal próprio, com limiar escolhido para triagem: no teste, encontra **33 dos 50 AVCs (recall de 0,66)** sinalizando 19% dos pacientes, com **ROC-AUC de 0,835** e PR-AUC de 0,214 (contra 0,049 do Dummy).

## Como navegar

| Página | O que tem |
|---|---|
| [Dataset](dataset.md) | origem dos dados, o que é cada coluna e os problemas de qualidade encontrados |
| [Análise exploratória](eda.md) | os achados das tarefas 1 a 3, com gráficos, tabelas e testes estatísticos |
| [Pré-processamento](preprocessing.md) | cada decisão da tarefa 4 e a justificativa baseada na análise |
| [Modelagem](modeling.md) | APS2, seção 1: métrica, os três modelos, busca de hiperparâmetros, mudança no pipeline, curva de aprendizado e limiar |
| [Avaliação no teste](evaluation.md) | APS2, seção 2: métricas, matriz de confusão e curva ROC do melhor modelo |
| [Interpretação](interpretation.md) | APS2, seção 3: coeficientes, importâncias e SHAP |
| [Discussão crítica](discussion.md) | APS2, seção 4: limitações do modelo e dos dados, operacionalização e referências |
