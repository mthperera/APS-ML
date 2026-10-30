# Discussão crítica

Seção 4 da APS2. Os números de apoio são calculados no [notebook 04](https://github.com/mthperera/APS-ML/blob/main/notebooks/04_evaluation.ipynb).

## Limitações do modelo

### 1. A avaliação no teste tem incerteza grande

O teste tem só 50 casos de AVC, e cada caso a mais ou a menos detectado muda o recall em 2 pontos. Um bootstrap (2.000 reamostragens do teste com reposição) dá os intervalos de confiança de 95%:

| Métrica | No teste | IC 95% |
|---|---|---|
| Recall | 0,660 | 0,529 a 0,787 |
| Precision | 0,174 | 0,121 a 0,228 |
| F1-score | 0,275 | 0,200 a 0,348 |
| ROC-AUC | 0,835 | 0,776 a 0,889 |
| PR-AUC | 0,214 | 0,142 a 0,331 |

O teste confirma que o modelo é muito melhor que o acaso (a ROC-AUC fica longe de 0,5), mas não tem poder para dizer que a Random Forest supera a regressão logística: o intervalo da PR-AUC contém a validação cruzada das duas (0,269 e 0,225). Para validar modelos de predição clínica, recomenda-se pelo menos 100 eventos, de preferência 200 (Collins et al., 2016); aqui há 50, e no treino só 199, abaixo do recomendado para desenvolver um modelo com esse número de features (Riley et al., 2020).

### 2. O ponto de operação só vale para a prevalência desta amostra

Recall e especificidade são propriedades do modelo, mas a precisão depende da proporção de AVC na população. Mantendo o recall (0,660) e a especificidade (0,838) do teste, pelo teorema de Bayes:

| Prevalência | Precisão | Alertas por caso de AVC |
|---|---|---|
| 4,9% (esta amostra) | 0,174 | 5,8 |
| 2% | 0,077 | 13 |
| 1% | 0,040 | 25 |
| 0,5% | 0,020 | 50 |
| 0,15% (incidência anual global) | 0,006 | 164 |

Na população geral, com cerca de 150 AVCs por 100 mil pessoas por ano (Feigin et al., 2021), o mesmo modelo emitiria mais de 160 alertas por caso. A prevalência de 4,9% mostra que a amostra não é uma população geral, e as probabilidades do modelo estão calibradas para ela, então ficariam superestimadas em outro contexto (Van Calster et al., 2019). Usar o modelo em outra população exige recalibrar as probabilidades e escolher de novo o limiar com dados dessa população.

## Limitações dos dados

### 1. Os dados são transversais: o target é histórico de AVC, não AVC futuro

Não há datas, então não se sabe se hipertensão, glicose, IMC ou tabagismo foram medidos antes ou depois do AVC. Quem teve AVC pode ter passado a tratar a pressão, mudado a dieta ou parado de fumar ("ex-fumante" pode ser consequência). O modelo aprende associações com quem *teve* AVC, não fatores que o *precedem*, e não pode ser usado como previsão de risco futuro. Para isso são necessários dados longitudinais, como os da coorte de Framingham, base dos escores clássicos de risco de AVC (Wolf et al., 1991).

O indicador de IMC ausente é um sintoma disso. Ele está entre as três features mais importantes em todos os métodos de interpretação, mas a falta de um peso medido não causa AVC: deve refletir como os dados foram coletados (por exemplo, pacientes internados após o AVC que não puderam ser pesados). Um modelo que se apoia num artefato de coleta perde esse sinal assim que o processo de coleta muda.

### 2. Faltam fatores de risco centrais, e a procedência é desconhecida

O estudo INTERSTROKE identificou dez fatores de risco modificáveis responsáveis por cerca de 90% do risco populacional de AVC: hipertensão, atividade física, relação ApoB/ApoA1, dieta, relação cintura-quadril, fatores psicossociais, tabagismo, causas cardíacas (como fibrilação atrial), álcool e diabetes (O'Donnell et al., 2016). O dataset tem versões grosseiras de poucos deles:

| Fator de risco | No dataset |
|---|---|
| hipertensão | só sim/não, sem a pressão medida |
| causas cardíacas | "doença cardíaca" genérica, sem distinguir fibrilação atrial |
| tabagismo | 30% desconhecido |
| diabetes | glicose média, sem diagnóstico |
| obesidade abdominal | IMC, em vez da relação cintura-quadril |
| colesterol, atividade física, dieta, álcool, estresse | ausentes |

Isso ajuda a explicar o teto de desempenho e os falsos negativos sem nenhum fator registrado. Além disso, o autor do dataset informa fonte confidencial: não se sabe país, período, critério de inclusão nem como o AVC foi diagnosticado, e portanto para qual população o modelo vale.

## Operacionalização e manutenção

### 1. Monitorar entradas e desempenho, com recalibração e re-treino programados

Em produção, a distribuição das entradas muda (outra população, novos protocolos de coleta) e o desempenho cai sem aviso; é um dos principais custos ocultos de sistemas de ML (Sculley et al., 2015). Na prática:

- comparar todo mês a distribuição de cada feature com a do treino, por testes univariados (Kolmogorov-Smirnov nas contínuas, qui-quadrado nas categóricas, com correção de Bonferroni), que Rabanser et al. (2019) mostraram ser um detector de mudança simples e eficaz. Atenção especial à taxa de IMC ausente, da qual o modelo depende;
- acompanhar a taxa de alertas (19% no teste): uma mudança brusca indica mudança na população ou no fluxo de dados antes mesmo de os desfechos chegarem;
- quando os desfechos reais chegarem, recalcular recall, precisão e calibração; abaixo de um recall mínimo acordado (por exemplo, 53%, o limite inferior do intervalo de confiança do teste), recalibrar ou retreinar com os dados novos.

### 2. Versionar o pipeline como artefato único, documentado, e com o limiar sob governança clínica

O objeto salvo pelo notebook 03 já reúne pré-processamento, modelo e limiar. Ele deve ser versionado junto com os dados e o código que o geraram, para que toda predição seja rastreável a uma versão. Cada versão deve ter um model card (Mitchell et al., 2019) com dados de treino, população em que vale, métricas com intervalos de confiança, desempenho por subgrupo (idade, gênero) e as limitações desta página.

O limiar não é detalhe técnico: define quantos pacientes são encaminhados, então deve ser definido com a equipe clínica conforme a capacidade de atendimento. Antes de afetar decisões, o modelo deve rodar em modo silencioso (as predições são registradas, mas não mostradas) para medir o desempenho na população real, e o relato deve seguir o checklist TRIPOD para modelos de predição clínica (Collins et al., 2015).

## Referências

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research*, 11, 2079–2107.
- Collins, G. S., Reitsma, J. B., Altman, D. G., & Moons, K. G. M. (2015). Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ*, 350, g7594.
- Collins, G. S., Ogundimu, E. O., & Altman, D. G. (2016). Sample size considerations for the external validation of a multivariable prognostic model: a resampling study. *Statistics in Medicine*, 35(2), 214–226.
- fedesoriano (2021). *Stroke Prediction Dataset*. Kaggle. https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
- Feigin, V. L. et al. (GBD 2019 Stroke Collaborators) (2021). Global, regional, and national burden of stroke and its risk factors, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. *The Lancet Neurology*, 20(10), 795–820.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.
- Lundberg, S. M., Erion, G., Chen, H. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.
- Mitchell, M. et al. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*, 220–229.
- O'Donnell, M. J. et al. (2016). Global and regional effects of potentially modifiable risk factors associated with acute stroke in 32 countries (INTERSTROKE): a case-control study. *The Lancet*, 388(10046), 761–775.
- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: an empirical study of methods for detecting dataset shift. *Advances in Neural Information Processing Systems*, 32.
- Riley, R. D. et al. (2020). Calculating the sample size required for developing a clinical prediction model. *BMJ*, 368, m441.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.
- Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28, 2503–2511.
- Strobl, C., Boulesteix, A.-L., Zeileis, A., & Hothorn, T. (2007). Bias in random forest variable importance measures: illustrations, sources and a solution. *BMC Bioinformatics*, 8, 25.
- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L., & Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. *BMC Medicine*, 17, 230.
- van Rijsbergen, C. J. (1979). *Information Retrieval* (2ª ed.). Butterworths.
- Wolf, P. A., D'Agostino, R. B., Belanger, A. J., & Kannel, W. B. (1991). Probability of stroke: a risk profile from the Framingham Study. *Stroke*, 22(3), 312–318.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
