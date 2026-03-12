import torch
from sklearn.metrics import roc_auc_score

class Evaluator:
    """
    Calcula métricas de qualidade da explicação comparando com o Ground-Truth.
    Agora inclui AUC-ROC conforme utilizado nos benchmarks do paper.
    """
    
    @staticmethod
    def calculate_auroc(pred_mask, gt_mask):
        """
        Calcula a Área sob a Curva ROC (AUC-ROC).
        Métrica robusta que independe de threshold (limiar) de corte.
        Ideal para avaliar se o explicador está ranqueando as arestas corretamente.
        """
        # Converter para numpy para usar sklearn
        y_true = gt_mask.detach().cpu().numpy()
        y_scores = pred_mask.detach().cpu().numpy()
        
        try:
            score = roc_auc_score(y_true, y_scores)
        except ValueError:
            # Caso extremo onde só existe uma classe no GT (ex: tudo zero)
            score = 0.5 
            
        return score
   
    @staticmethod
    def calculate_jaccard_accuracy(pred_mask, gt_mask, threshold=0.5):
        """
        Calcula a Acurácia da Explicação (GEA) usando o Índice de Jaccard.
        Fórmula: |Interseção| / |União|
        """
        # 1. Binarizar a máscara predita (soft probabilities -> 0 ou 1)
        # O paper sugere tratar atributos como 0 ou 1 
        pred_binary = (pred_mask > threshold).float()
        gt_binary = gt_mask.float()
        
        # 2. Calcular Interseção e União
        intersection = (pred_binary * gt_binary).sum()
        union = pred_binary.sum() + gt_binary.sum() - intersection
        
        # Evitar divisão por zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
            
        jaccard = intersection / union
        return jaccard.item()
   
   
    @staticmethod
    def calculate_recall(pred_mask, gt_mask, threshold=0.5):
        """
        Calcula quanto do Ground-Truth foi recuperado.
        Útil para saber se o explainer achou a casa toda ou só um pedaço.
        """
        pred_binary = (pred_mask > threshold).float()
        gt_binary = gt_mask.float()
        
        true_positives = (pred_binary * gt_binary).sum()
        possible_positives = gt_binary.sum()
        
        if possible_positives == 0:
            return 1.0 # Não tinha nada para achar e não achou nada
            
        return (true_positives / possible_positives).item()
   
    @staticmethod
    def calculate_unfaithfulness(model, data, node_idx, edge_mask, threshold=0.5):
        """
        Calcula a Graph Explanation Faithfulness (GEF) conforme Eq. 3 do artigo GraphXAI.
        Mede a INFIDELIDADE: Quanto menor, mais fiel a explicação é ao modelo.
        """
        model.eval()
        
        # 1. Predição Original (f(S_u))
        with torch.no_grad():
            out_original = model(data.x, data.edge_index)
            # Pegamos as probabilidades (exp porque o modelo retorna log_softmax)
            probs_original = torch.exp(out_original[node_idx])

        # 2. Criar Subgrafo Explicado (S_u')
        # Filtramos arestas importantes baseadas no threshold
        mask_bool = edge_mask > threshold
        subset_edge_index = data.edge_index[:, mask_bool]
        
        # Se a explicação cortou tudo (nenhuma aresta), a fidelidade é péssima (máxima infidelidade)
        if subset_edge_index.size(1) == 0:
            return 1.0

        # 3. Predição com a Explicação (f(S_u'))
        with torch.no_grad():
            # Passamos o grafo original (X) mas com arestas filtradas
            out_explained = model(data.x, subset_edge_index)
            probs_explained = torch.exp(out_explained[node_idx])

        # 4. Calcular KL Divergence
        # KL(P || Q) = sum(P * log(P/Q))
        # Adicionamos epsilon para evitar log(0)
        epsilon = 1e-8
        kl_div = torch.sum(probs_original * torch.log((probs_original + epsilon) / (probs_explained + epsilon)))
        
        # 5. Fórmula do GraphXAI (Eq. 3)
        # GEF = 1 - exp(-KL)
        gef = 1 - torch.exp(-kl_div)
        
        return gef.item()
    @staticmethod
    def calculate_fidelity(model, data, node_idx, edge_mask, threshold=0.5):
        model.eval()
        
        with torch.no_grad():
            out_orig = model(data.x, data.edge_index)
            pred_class = out_orig[node_idx].argmax(dim=-1).item()
            p_orig = torch.exp(out_orig[node_idx, pred_class]).item()
            
        # Fidelity+ (Apenas a explicação)
        mask_plus = edge_mask > threshold
        edge_index_plus = data.edge_index[:, mask_plus]
        
        with torch.no_grad():
            if edge_index_plus.size(1) == 0:
                p_plus = 0.0
            else:
                out_plus = model(data.x, edge_index_plus)
                p_plus = torch.exp(out_plus[node_idx, pred_class]).item()
                
        # Fidelity- (Grafo sem a explicação)
        mask_minus = edge_mask <= threshold
        edge_index_minus = data.edge_index[:, mask_minus]
        
        with torch.no_grad():
            if edge_index_minus.size(1) == 0:
                p_minus = 0.0
            else:
                out_minus = model(data.x, edge_index_minus)
                p_minus = torch.exp(out_minus[node_idx, pred_class]).item()
                
        fid_plus = p_plus 
        fid_minus = max(0.0, p_orig - p_minus)
        
        return fid_plus, fid_minus