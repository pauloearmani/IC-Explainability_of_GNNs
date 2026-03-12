# src/explainers.py

import torch
from torch_geometric.explain import Explainer, GNNExplainer


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
    Wrapper para o GNNExplainer usando a API atual do PyG (torch_geometric.explain.Explainer).

    A API antiga chamava GNNExplainer diretamente como função — isso foi removido.
    A API nova exige encapsular o GNNExplainer dentro de um Explainer, que cuida
    de toda a lógica de target, mascaramento e normalização.
    """
    def __init__(self, model, epochs=200, edge_size=0.005, edge_ent=1.0):
        super().__init__(model)

        # API correta: GNNExplainer vai dentro de Explainer
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type='model',       # explica a predição do modelo (não o GT)
            node_mask_type='attributes',    # máscara por feature de nó
            edge_mask_type='object',        # máscara por aresta (o que nos interessa)
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',    # nosso modelo retorna log_softmax
            ),
        )

    def explain_node(self, node_idx, data, **kwargs):
        """
        Gera a explicação para um nó específico.

        Retorna:
            edge_mask (Tensor): importância de cada aresta para a predição do nó.
                                Shape: [num_edges], valores em [0, 1].
        """
        # A API nova descobre o target automaticamente a partir da predição do modelo
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=node_idx,
        )

        # explanation.edge_mask: scores de importância por aresta
        return explanation.edge_mask

    def explain_all_motif_nodes(self, data):
        """
        Explica todos os nós de motif (classe 1) e agrega as máscaras.

        Retorna:
            all_masks (dict): {node_idx: edge_mask} para cada nó de motif
            agg_mask (Tensor): média das máscaras sobre todos os nós explicados
        """
        motif_indices = (data.y == 1).nonzero(as_tuple=True)[0].tolist()
        print(f"Explicando {len(motif_indices)} nós de motif...")

        all_masks = {}
        agg_mask = torch.zeros(data.num_edges)

        for i, node_idx in enumerate(motif_indices):
            if i % 10 == 0:
                print(f"  Nó {i+1}/{len(motif_indices)}...")
            edge_mask = self.explain_node(node_idx, data)
            all_masks[node_idx] = edge_mask.detach().cpu()
            agg_mask += edge_mask.detach().cpu()

        # Normaliza a máscara agregada para [0, 1]
        if agg_mask.max() > 0:
            agg_mask = agg_mask / agg_mask.max()

        print(f"Concluído! {len(motif_indices)} explicações geradas.")
        return all_masks, agg_mask