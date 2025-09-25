# src/explainers.py

import torch
from torch_geometric.explain import GNNExplainer # A importação está correta

class ExplainerModule:
    """
    Classe base para garantir que todos os explainers tenham a mesma interface.
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def explain_node(self, node_idx, data, **kwargs):
        raise NotImplementedError("Este método deve ser implementado pela subclasse.")


class GNNExplainerWrapper(ExplainerModule):
    """
    Implementação modular para o GNNExplainer, com a correção do argumento 'target'.
    """
    def __init__(self, model, epochs=200):
        super().__init__(model)
        self.explainer = GNNExplainer(epochs=epochs)

    def explain_node(self, node_idx, x, edge_index, **kwargs):
        """
        Gera a explicação para um nó específico.
        """
        print(f"Gerando explicação para o nó {node_idx} com GNNExplainer...")

        # CORREÇÃO: Antes de explicar, precisamos saber qual foi a predição do modelo
        # para usá-la como o 'target' da explicação.
        with torch.no_grad():
            prediction_logits = self.model(x, edge_index)
            # .argmax() encontra o índice da maior log-probabilidade, que é a classe predita.
            predicted_class = prediction_logits[node_idx].argmax().item()

        # Chamamos o explicador como uma função, passando o modelo, os dados,
        # e o 'target' que acabamos de descobrir.
        explanation = self.explainer(
            self.model,
            x,
            edge_index,
            index=node_idx,
            target=torch.tensor([predicted_class]) # Passando a classe predita como o alvo
        )

        # A API retorna um objeto 'Explanation' que contém as máscaras
        return explanation.node_feat_mask, explanation.edge_mask