import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_networkx

def split_train_test(data, test_size=0.2, random_state=42):
    """
    Divide os nós do grafo em máscaras de treino e teste, 
    mantendo a proporção correta (estratificada) das classes.
    """
    num_nodes = data.num_nodes
    indices = list(range(num_nodes))
    
    # Stratify garante que a proporção de motifs vs base seja igual no treino e teste
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=data.y.cpu().numpy()
    )
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.test_mask = test_mask
    
    return data

def compute_class_weights(data, device):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe.
    Resolve o problema de desbalanceamento entre nós base e nós de motif.
    
    Exemplo: se há 75% classe 0 e 25% classe 1, os pesos serão [0.33, 1.0]
    fazendo o modelo penalizar 3x mais os erros na classe 1 (motif).
    """
    labels = data.y.cpu().numpy()
    classes, counts = np.unique(labels, return_counts=True)
    # Peso = 1 / frequência, normalizado pelo peso máximo
    weights = 1.0 / counts.astype(float)
    weights = weights / weights.max()
    print(f"  → Classes: {classes}, Contagens: {counts}, Pesos: {np.round(weights, 4)}")
    return torch.tensor(weights, dtype=torch.float).to(device)

def train_gcn_model(model, data, optimizer, loss_fn=None, epochs=1501, eval_interval=50, device='cpu'):
    """
    Treina a GCN e avalia métricas completas.
    Salva e retorna o melhor modelo com base no F1-Score da Classe 1 (Motif).
    
    CORREÇÃO: loss_fn agora é criada internamente com class_weight automático
    se não for fornecida externamente. Isso resolve o colapso para classe majoritária.
    """
    # --- CORREÇÃO 1: Loss com class_weight automático ---
    if loss_fn is None:
        class_weights = compute_class_weights(data, device)
        loss_fn = torch.nn.NLLLoss(weight=class_weights)
        print(f"  → Usando NLLLoss com pesos automáticos de classe.")
    
    best_f1 = 0.0
    best_metrics = {}
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Agora vamos ver a Loss e a Acurácia também!
    print(f"{'Época':<6} | {'Loss':<7} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                pred_logits = model(data.x, data.edge_index)
                pred_classes = pred_logits.argmax(dim=1).cpu().numpy()
                y_true = data.y.cpu().numpy()
                test_mask = data.test_mask.cpu().numpy()
                
                y_test = y_true[test_mask]
                pred_test = pred_classes[test_mask]
                
                acc = (y_test == pred_test).mean()
                precision = precision_score(y_test, pred_test, average='binary', zero_division=0)
                recall = recall_score(y_test, pred_test, average='binary', zero_division=0)
                f1 = f1_score(y_test, pred_test, average='binary', zero_division=0)
                
                print(f"{epoch:<6} | {loss.item():.4f}  | {acc:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f}")
                
                # Critério de sucesso continua sendo o F1-Score
                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = {'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1}
                    best_model_wts = copy.deepcopy(model.state_dict())
    
    if best_f1 > 0:
        model.load_state_dict(best_model_wts)
    
    print("-" * 55)
    print(f"🏆 Melhor Teste -> Acc: {best_metrics.get('acc', 0):.4f} | Prec: {best_metrics.get('precision', 0):.4f} | Rec: {best_metrics.get('recall', 0):.4f} | F1: {best_f1:.4f}")
    
    return model, best_metrics

def visualizar_grafo_gt(data, title="Visualização do Grafo (Motif em Vermelho)"):
    """
    Plota o grafo destacando os nós e arestas que pertencem ao Motif (Ground Truth).
    """
    G = to_networkx(data, to_undirected=True)
    node_colors = ['red' if y == 1 else '#a0cbe2' for y in data.y]
    
    plt.figure(figsize=(12, 10))
    pos = nx.kamada_kawai_layout(G)
    
    # Desenhar os nós
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100)
    
    gt_edges = []
    normal_edges = []
    
    edge_index = data.edge_index.t().tolist()
    if hasattr(data, 'edge_mask_gt'):
        gt_mask = data.edge_mask_gt.tolist()
        
        gt_edge_set = set()
        for (u, v), is_gt in zip(edge_index, gt_mask):
            if is_gt == 1.0:
                gt_edge_set.add(tuple(sorted((u, v))))
                
        for u, v in G.edges():
            if tuple(sorted((u, v))) in gt_edge_set:
                gt_edges.append((u, v))
            else:
                normal_edges.append((u, v))
    else:
        normal_edges = list(G.edges())
            
    # Desenhar arestas normais mais transparentes
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, alpha=0.2)
    # Desenhar arestas do motif mais grossas e vermelhas
    if gt_edges:
        nx.draw_networkx_edges(G, pos, edgelist=gt_edges, edge_color='red', width=3)
    
    plt.title(title)
    plt.axis('off')
    plt.show()