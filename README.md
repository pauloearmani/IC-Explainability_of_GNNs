# Explicabilidade de GNNs para Predição de Séries Temporais

## Resumo (Abstract)
O presente projeto tem por objetivo explicar modelos de aprendizado profundo (deep learning) para predição de séries temporais, utilizando Redes Neurais de Grafos (Graph Neural Networks - GNNs). Será utilizado como estudo de caso um modelo de predição de séries temporais de COVID-19 no Brasil, composto de Redes Convolucionais de Grafo (GCN) em conjunto com Memória de curto e longo prazo (LSTM), tendo como entradas séries temporais de casos de COVID-19 nas cidades brasileiras e a rede de mobilidade humana entre elas. O principal objetivo é utilizar técnicas de explicabilidade como GNNExplainer, SHAP e LIME para compreender quais cidades vizinhas mais influenciam na predição dos casos de cada cidade.

## Status do Projeto
- Fase 1 Concluída: Framework de análise e avaliação de explicabilidade (GNNExplainer) finalizado em dataset de benchmark (Cora).

## Instalação
1. Clone o repositório: `git clone https://github.com/seu-usuario/seu-repositorio.git`
2. Navegue até a pasta do projeto: `cd seu-repositorio`
3. Crie um ambiente virtual: `python -m venv venv`
4. Ative o ambiente: `source .venv/bin/activate` (ou `.venv\Scripts\activate` no Windows)
5. Instale as dependências: `pip install -r requirements.txt`

## Como Usar
O projeto é dividido em notebooks Jupyter para facilitar a experimentação e a visualização dos resultados.
1. Treinamento do Modelo Base:
- Para treinar o modelo GCN no dataset Cora e salvar o arquivo de pesos (`gcn_cora.pth`), execute todas as células do notebook: `notebooks/01-treinamento_gcn_cora.ipynb`.
2. Análise de Explicabilidade:
- Para carregar o modelo treinado, gerar explicações para um nó de teste usando GNNExplainer, visualizar o resultado e calcular as métricas de avaliação (Fidelidade e Concisão), execute o notebook: `notebooks/02-teste_explicador.ipynb`.

## Estrutura do Repositório
O projeto segue uma estrutura modular para separar responsabilidades e facilitar a manutenção e expansão do código.
```
├── notebooks/          # Contém os notebooks Jupyter para experimentação, treinamento e análise.
│   ├── 01-treinamento_gcn_cora.ipynb
│   └── 02-teste_explicador.ipynb
│
├── src/                # Código fonte principal, organizado em módulos.
│   ├── models.py       # Definição das arquiteturas das redes neurais (ex: GCN).
│   ├── explainers.py   # Implementação dos wrappers dos métodos de explicação (ex: GNNExplainer).
│   └── metrics.py      # Funções para calcular as métricas de avaliação (Fidelidade, Concisão).
│
├── gcn_cora.pth        # Arquivo de pesos do modelo GCN treinado no dataset Cora.
├── requirements.txt    # Lista de dependências Python para o projeto.
└── README.md           # Este arquivo de documentação.
```

## Referências e Citações
- [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)
- [A Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)

## Autor e Contato
- **Nome:** Paulo Eduardo Costalonga Armani
- **LinkedIn:** [www.linkedin.com/in/pauloeduardoarmani]
- **GitHub:** [https://github.com/pauloearmani]
