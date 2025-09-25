# src/metrics.py
import torch

def calculate_sparsity(edge_mask, threshold=None):
    """
    Calcula a Concisão (Sparsity) de uma explicação.

    Args:
        edge_mask (Tensor): A máscara de importância para as arestas.
        threshold (float, optional): O limiar para considerar uma aresta importante.
                                     Se None, a média da máscara será usada.

    Returns:
        float: A proporção de arestas consideradas não importantes (quanto maior, mais concisa).
    """
    if threshold is None:
        threshold = edge_mask.mean()
    
    num_unimportant_edges = (edge_mask <= threshold).sum().item()
    total_edges = edge_mask.numel()
    
    return num_unimportant_edges / total_edges if total_edges > 0 else 0.0


def calculate_fidelity(model, node_idx, data, edge_mask, fidelity_type='+'):
    """
    Calcula a Fidelidade (Fidelity+ ou Fidelity-) de forma numericamente estável,
    trabalhando diretamente com log-probabilidades.
    """
    model.eval()
    
    with torch.no_grad():
        # 1. Obter a log-probabilidade original para a classe predita
        original_log_probs = model(data.x, data.edge_index)
        predicted_class = original_log_probs[node_idx].argmax(dim=-1)
        # Pegamos o valor diretamente em log, sem .exp()
        original_log_prob_for_class = original_log_probs[node_idx, predicted_class].item()

        # 2. Criar o grafo perturbado baseado na máscara
        threshold = edge_mask.mean()
        
        if fidelity_type == '+': # Manter apenas arestas importantes
            important_edges_mask = edge_mask > threshold
            perturbed_edge_index = data.edge_index[:, important_edges_mask]
        elif fidelity_type == '-': # Remover arestas importantes
            unimportant_edges_mask = edge_mask <= threshold
            perturbed_edge_index = data.edge_index[:, unimportant_edges_mask]
        else:
            raise ValueError("fidelity_type deve ser '+' ou '-'")

        # 3. Fazer a nova predição com o grafo perturbado
        perturbed_log_probs = model(data.x, perturbed_edge_index)
        # Pegamos o novo valor diretamente em log, sem .exp()
        perturbed_log_prob_for_class = perturbed_log_probs[node_idx, predicted_class].item()

    # A fidelidade é a queda na log-probabilidade.
    # Um número positivo grande aqui significa uma grande queda de confiança.
    return original_log_prob_for_class - perturbed_log_prob_for_class