# Explicabilidade de GNNs para Predição de Séries Temporais

## Resumo

Este projeto investiga mecanismos de explicabilidade de modelos de Redes Neurais de Grafos (GNNs), com foco no método GNNExplainer e na avaliação sistemática da qualidade das explicações geradas. O estudo de caso final é um modelo GCN-LSTM de predição de séries temporais de COVID-19 em municípios brasileiros, utilizando o grafo de mobilidade humana como estrutura relacional.

O projeto é desenvolvido como Iniciação Científica na Universidade Federal de Ouro Preto (UFOP).

---

## Status

| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | GNNExplainer no dataset Cora (benchmark real) | ✅ Concluída |
| 2 | Avaliação com grafos sintéticos e ground-truth conhecido | ✅ Concluída |
| 3 | Métodos alternativos (SHAP, SubgraphX) no benchmark sintético | 🔄 Planejada |
| 4 | Extensão ao modelo GCN-LSTM (COVID-19) | 🔄 Planejada |

---

## Resultados Principais

### Fase 1 — Dataset Cora
- GCN 2 camadas treinada no Cora para classificação de nós (7 classes)
- GNNExplainer avaliado em nós de teste representativos
- **Fidelity+ = 1.0, Fidelity- = 0.0** — explicações suficientes mas não necessárias
- Interpretação ambígua: ausência de ground-truth estrutural impede distinguir limitação do explicador de complexidade do dataset

### Fase 2 — Grafos Sintéticos com Ground-Truth

Gerador próprio com motifs plantados (house 5 nós, star 6 nós), grafo base Barabási-Albert.

| Motif | F1 Modelo | AUC-ROC Expl. | Jaccard | Recall | Fidelity+ | Fidelity- | Unfaithfulness |
|-------|-----------|----------------|---------|--------|-----------|-----------|----------------|
| House | 0.9744 | 0.7597 | 0.7597 | 0.8167 | 0.9436 | 0.3874 | 0.0919 |
| Star  | 0.9362 | 0.7082 | 0.7082 | 0.8250 | 0.9914 | 0.4250 | 0.1229 |

A Fidelity- não-nula nos sintéticos (vs. 0.0 no Cora) confirma que o comportamento observado na Fase 1 é parcialmente atribuível à complexidade estrutural do dataset real — e não apenas a uma limitação do explicador.

---

## Estrutura do Repositório

```
├── notebooks/
│   ├── 01_treinamento_gcn_cora.ipynb      # Fase 1 — treino da GCN no Cora
│   ├── 02_teste_explicador_cora.ipynb     # Fase 1 — GNNExplainer no Cora + métricas
│   ├── 03_pipeline_sintetico.ipynb        # Fase 2 — pipeline completo com grafos sintéticos
│   └── 04_experimentos_finais.ipynb       # Fase 2 — resultados e análise comparativa
│
├── src/
│   ├── generators.py   # Gerador de grafos sintéticos com motifs (house, star, cycle)
│   ├── models.py       # Arquiteturas GCN (2 e 3 camadas)
│   ├── explainers.py   # Wrapper do GNNExplainer com correção de target
│   ├── evaluator.py    # Métricas: AUC-ROC, Jaccard, Recall, Unfaithfulness, Fidelity
│   ├── metrics.py      # Funções auxiliares de métricas (Sparsity, Fidelity log-prob)
│   └── utils.py        # Treinamento, split estratificado, visualização de grafos
│
├── results/
│   └── models/
│       └── gcn_cora.pth   # Pesos da GCN treinada no Cora
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
git clone https://github.com/pauloearmani/IC-Explainability_of_GNNs.git
cd IC-Explainability_of_GNNs
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Nota:** Este projeto requer PyTorch e PyTorch Geometric. Certifique-se de instalar a versão compatível com seu CUDA. Consulte [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) para instruções específicas.

---

## Como Usar

### Pipeline sintético completo (Fase 2)
```bash
jupyter notebook notebooks/03_pipeline_sintetico.ipynb
jupyter notebook notebooks/04_experimentos_finais.ipynb
```

### Reproduzir Fase 1 (Cora)
```bash
jupyter notebook notebooks/01_treinamento_gcn_cora.ipynb
jupyter notebook notebooks/02_teste_explicador_cora.ipynb
```

### Usar os módulos diretamente

```python
from src.generators import SyntheticGraphGenerator
from src.models import GCNClassifier
from src.evaluator import Evaluator

# Gerar grafo com motif house
gen = SyntheticGraphGenerator(num_nodes=300, num_houses=20, motif_type='house')
data = gen.generate()

# Treinar modelo
model = GCNClassifier(num_features=10, num_classes=2, hidden_dim=64)
```

---

## Metodologia

O projeto segue uma sequência deliberada de experimentos, motivada pela literatura recente:

1. **Avaliação em dado real (Cora):** revelou ambiguidade interpretativa causada pela ausência de ground-truth estrutural — Fidelity- = 0.0 não distingue limitação do explicador de complexidade do dataset.

2. **Experimentos sintéticos controlados:** motivados por [Miró-Nicolau et al. (2025)](#referências), que demonstra que métricas de fidelidade podem ser não confiáveis em modelos não-lineares, e por [Agarwal et al. (2023)](#referências), que documenta degradação sistemática de explicadores ao passar de sintéticos para dados reais.

3. **Extensão ao estudo de caso final (GCN-LSTM + COVID-19):** contexto sem ground-truth estrutural, onde as lições das fases anteriores serão diretamente aplicadas.

---

## Referências

- Ying et al. **GNNExplainer: Generating Explanations for Graph Neural Networks.** NeurIPS, 2019. [arxiv](https://arxiv.org/abs/1903.03894)
- Kipf & Welling. **Semi-Supervised Classification with Graph Convolutional Networks.** ICLR, 2017. [arxiv](https://arxiv.org/abs/1609.02907)
- Agarwal et al. **Evaluating explainability for graph neural networks.** Scientific Data, v.10, p.144, 2023. [doi](https://doi.org/10.1038/s41597-023-01974-x)
- Miró-Nicolau et al. **A comprehensive study on fidelity metrics for XAI.** Information Processing and Management, v.62, p.103900, 2025. [doi](https://doi.org/10.1016/j.ipm.2024.103900)

---

## Autor

**Paulo Eduardo Costalonga Armani**  
Iniciação Científica — UFOP  
[linkedin.com/in/pauloeduardoarmani](https://linkedin.com/in/pauloeduardoarmani) · [github.com/pauloearmani](https://github.com/pauloearmani)