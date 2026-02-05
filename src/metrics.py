"""
Metrics to evaluate SAEs. Includes:

- R^2, L2 :
    - reconstruction error
- L0, L1 :
    - sparsity
- dead features
    - "The Dead Codes measure the fraction of dictionary atoms that remain unused across the dataset, highlighting inefficiencies in the learned representation" - archetypal SAE
    - not sure what this means but ok
- C-insertion, C-deletion, C-$\mu$fidelity
    - How 
- stability
    - Wasserstein distance - Monge (Permutation), weighted or not, Kantorovich plan
    - Sinkhorn distance
    - is the learned representation stable?
- OOD score
    - How far is each feature vector from the closest embedding? Again, mean vs weighted mean?
- rank
    - stable
        - smooth proxy for rank estimation in high dimensions - numerical errors : all dictionaries are nearly full rank
    - effective
        - entropy of the singular value distribution
        - "An effective rank close to k suggests that all dictionary atoms are equally important, whereas a low effective rank implies that only a few dominant concepts capture most of the variation in the data"
- coherence
    - "quantifies redundancy between dictionary atoms by measuring the maximum pairwise cosine similarity"
- connectivity
    - "measures the diversity of concept usage by counting the number of unique co-activations within the code matrix"
- negative interference
    - "quantifies the extent to which co-activated concepts cancel each other out, reducing their effectiveness"
- feature linearity
    - Compare codes and pre code activations - if similar activation patterns, then linear.
"""

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from overcomplete.metrics import r2_score

from scipy.optimize import linear_sum_assignment
import ot

##########
# General quality metrics
##########

@torch.no_grad()
def l2_r2(sae, data_loader, return_r2=True, return_l2=True):
    # reconstruction error
    sae.eval()

    img_l2 = 0
    img_r2 = 0
    text_l2 = 0
    text_r2 = 0

    n_images = 0
    n_texts = 0
    
    for image_features, text_features in tqdm(data_loader):
        n_images += image_features.shape[0]
        n_texts += text_features.shape[0]

        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the reconstruction
        _, _, img_hat = sae(image_features)
        _, _, text_hat = sae(text_features)

        # compute the reconstruction error
        img_l2 += F.mse_loss(img_hat, image_features, reduction='sum').item()
        text_l2 += F.mse_loss(text_hat, text_features, reduction='sum').item()

        img_r2 += r2_score(image_features, img_hat) * image_features.shape[0]
        text_r2 += r2_score(text_features, text_hat) * text_features.shape[0]

    l2 = (img_l2 + text_l2) / (n_images + n_texts)
    r2 = (img_r2 + text_r2) / (n_images + n_texts)

    if return_l2 and return_r2:
        return l2, r2
    elif return_l2:
        return l2
    elif return_r2:
        return r2
    else:
        raise ValueError("At least one of return_l2 or return_r2 must be True")

@torch.no_grad()
def l0_l1(sae, data_loader, return_l0=True, return_l1=True):
    # sparsity of the codes
    sae.eval()

    l0 = 0
    l1 = 0
    n_images = 0
    n_texts = 0

    for image_features, text_features in tqdm(data_loader):
        n_images += image_features.shape[0]
        n_texts += text_features.shape[0]

        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, _ = sae(image_features)
        _, z_text, _ = sae(text_features)
        # compute the sparsity
        l0 += torch.sum(z_img != 0).item() + torch.sum(z_text != 0).item()
        l1 += torch.sum(torch.abs(z_img)).item() + torch.sum(torch.abs(z_text)).item()

    l0 /= (n_images + n_texts)
    l1 /= (n_images + n_texts)

    if return_l0 and return_l1:
        return l0, l1
    elif return_l0:
        return l0
    elif return_l1:
        return l1
    else:
        raise ValueError("At least one of return_l0 or return_l1 must be True")

@torch.no_grad()
def dead_features(sae, data_loader):
    # fraction of dictionary atoms that remain unused across the dataset
    # "The Dead Codes measure the fraction of dictionary atoms that remain unused across the dataset, highlighting inefficiencies in the learned representation" - archetypal SAE
    sae.eval()

    dead_mask = torch.zeros(sae.nb_concepts, device=sae.device)

    for image_features, text_features in tqdm(data_loader):
        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, _ = sae(image_features)
        _, z_text, _ = sae(text_features)

        # compute the dead features
        dead_mask += (z_img != 0).sum(dim=0) + (z_text != 0).sum(dim=0)
    
    dead_mask = (dead_mask == 0).float()

    dead_count = dead_mask.sum().item()
    total_count = dead_mask.numel()

    return dead_count / total_count

@torch.no_grad()
def __IG(sae, codes, x, metric='l2', steps=10):
    assert metric in ['l2', 'R2'], f"metric must be 'l2' or 'R2'. Got {metric} instead"
    
    # Integrated Gradients : int(0, 1) delta gamma(t) / delta gamma_i(t) * delta gamma_i(t) / delta t dt
    # linear interpolation between 0 and codes, codes.grad w.r.t. metric(x, x_hat)
    # final IG attr simplifies to 1/steps codes @ sum_i{code_interpol.grad}

    grad_sum = torch.zeros_like(codes)
    for step in range(1, steps + 1):
        alpha = step / steps
        with torch.enable_grad():
            codes_interpol = codes * alpha # shape (N, nb_concepts)
            codes_interpol.requires_grad_()
            x_hat = sae.decode(codes_interpol) # shape (N, d)
            if metric == 'l2':
                loss = F.mse_loss(x_hat, x)
            elif metric == 'R2':
                loss = 1-r2_score(x, x_hat)
            else:
                raise ValueError(f"metric {metric} not supported")
            grad = torch.autograd.grad(loss, codes_interpol)[0] # shape (N, nb_concepts)
            grad_sum += grad

    effect = - 1/steps * (grad_sum * codes) # shape (N, nb_concepts)

    return effect

def __C_intervention(sae, x, x_hat, z, attr, perm, k, metric='l2', complement=False):
    # keep top k features sorted by attr
    """
    Metric behavior :
    L2 : (||x - x_hat_k||_2 - ||x||_2) / (||x - x_hat||_2 - ||x||_2)
        -> 0 if x_hat_k = 0,
           € [0, 1] if x_hat_k = mean(x),
           1 if x_hat_k = x_hat,
           > 1 if x_hat_k = x
    R2 : R2(x_hat_k, x) / R2(x_hat, x) -> simplifies to (||x - x_hat_k||_2^2 - ||x - x_mean||_2^2) / (||x - x_hat||_2^2 - ||x - x_mean||_2^2)
        -> < 0 if x_hat_k = 0,
           0 if x_hat_k = mean(x),
           1 if x_hat_k = x_hat,
           > 1 if x_hat_k = x
    """
    if len(perm.shape) == 1:
        perm = perm.unsqueeze(0) # shape (1, nb_concepts)
        perm = perm.expand(z.shape[0], -1) # shape (N, nb_concepts)
        attr = attr.unsqueeze(0) # shape (1, nb_concepts)
        attr = attr.expand(z.shape[0], -1) # shape (N, nb_concepts)

    assert perm.shape[0] == z.shape[0], f"perm shape {perm.shape} and z shape {z.shape} must be the same"

    K_th = attr[torch.arange(attr.shape[0]).cpu(), perm[:, k - 1]] # shape (N, )
    max_elt = torch.max(attr, dim=1)[0] # shape (N, )
    if k == 0:
        K_th = max_elt + 1
    mask = attr >= K_th.unsqueeze(1) if not complement else attr < K_th.unsqueeze(1) # shape (N, nb_concepts)

    z_k = torch.zeros_like(z) # shape (N, nb_concepts)
    z_k[mask] = z[mask] # shape (N, nb_concepts)

    x_hat_k = sae.decode(z_k) # shape (N, D)

    if metric == 'l2':
        # return (||x - x_hat_k||_2 - ||x||_2) / (||x - x_hat||_2 - ||x||_2)
        x_norm = torch.linalg.norm(x, ord=2, dim=1) # shape (N, )
        x_hat_k_dist = torch.linalg.norm(x - x_hat_k, ord=2, dim=1) # shape (N, )
        x_hat_dist = torch.linalg.norm(x - x_hat, ord=2, dim=1) # shape (N, )
        return ((x_hat_k_dist - x_norm) / (x_hat_dist - x_norm)).sum().item() # shape (1, )
    if metric == 'R2':
        return r2_score(x, x_hat_k).mean().item() * x.shape[0] # shape (1, )

@torch.no_grad()
def _C_insertion_or_deletion(sae, data_loader, complement, L0=None, local=True, metric='R2', steps=10, return_curve=False):
    # C-insertion
    """
    if a L0 is provided, K = L0
    else K = nb_concepts

    Local :
    for k in range(1, K):
        for x in dataset:
            measure attr
            keep top k features sorted by attr
            measure metric
        Aggregate metric faithfulness

    Global :
    for x in dataset:
        measure attr
    aggregate attr
    for k in range(1, K):
        for x in dataset:
            keep top k features sorted by attr
            measure metric
        Aggregate metric faithfulness
    
    return faithfulness vs k and final score : area under the curve (higher is better) (~ mean_k{faithfulness[k]})
    """
    assert metric in ['l2', 'R2'], f"metric must be 'l2' or 'R2'. Got {metric} instead"

    K = L0 if L0 is not None else sae.nb_concepts
    faithfulnesses = [0 for _ in range(0, K)]
    n_inputs = 0

    if not local:
        attr = torch.zeros(sae.nb_concepts, device=sae.device)
    
    for image_features, text_features in tqdm(data_loader):
        for x in [image_features, text_features]:
            n_inputs += x.shape[0] # N
            x = x.to(sae.device) # shape (N, D)
            _, z, x_hat = sae(x) # shape (N, nb_concepts), shape (N, D)

            attr_x = __IG(sae, z, x, metric=metric, steps=steps) # shape (N, nb_concepts)

            if local:
                perm = torch.argsort(attr_x, dim=1, descending=True).cpu() # shape (N, nb_concepts)

                for k in range(0, K):
                    f = __C_intervention(sae, x, x_hat, z, attr_x, perm, k, metric=metric, complement=complement)
                    faithfulnesses[k] += f
                    # faithfulnesses[k] += __C_intervention(sae, x, x_hat, z, attr_x, perm, k, metric=metric, complement=complement)
            else:
                attr += attr_x.sum(dim=0)
            
    if not local:
        attr /= n_inputs
        perm = torch.argsort(attr, descending=True) # shape (nb_concepts, )

        for image_features, text_features in tqdm(data_loader):
            for x in [image_features, text_features]:
                x = x.to(sae.device)
                _, z, x_hat = sae(x)
                for k in range(0, K):
                    faithfulnesses[k] +=  __C_intervention(sae, x, x_hat, z, attr, perm, k, metric=metric, complement=complement)

    score = 0
    for k in range(0, K):
        faithfulnesses[k] /= n_inputs
        score += faithfulnesses[k]
    score /= (K + 1)

    if return_curve:
        return score, faithfulnesses
    return score

@torch.no_grad()
def C_insertion(sae, data_loader, L0=None, local=True, metric='R2', steps=10, return_curve=False):
    return _C_insertion_or_deletion(sae, data_loader, complement=False, L0=L0, local=local, metric=metric, steps=steps, return_curve=return_curve)

@torch.no_grad()
def C_deletion(sae, data_loader, L0=None, local=True, metric='R2', steps=10, return_curve=False):
    return _C_insertion_or_deletion(sae, data_loader, complement=True, L0=L0, local=local, metric=metric, steps=steps, return_curve=return_curve)

@torch.no_grad()
def C_mu_fidelity(sae, data_loader, L0=None, local=True, metric='R2', steps=10, return_curve=False):
    # C-$\mu$fidelity
    ...

@torch.no_grad()
def l0_l1_l2_r2_dead(sae, data_loader):
    sae.eval()

    img_l2 = 0
    img_r2 = 0
    text_l2 = 0
    text_r2 = 0

    l0 = 0
    l1 = 0
    
    dead_mask = torch.zeros(sae.nb_concepts, device=sae.device)

    n_images = 0
    n_texts = 0
    
    for image_features, text_features in tqdm(data_loader):
        n_images += image_features.shape[0]
        n_texts += text_features.shape[0]

        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, img_hat = sae(image_features)
        _, z_text, text_hat = sae(text_features)
        
        # compute the sparsity
        l0 += torch.sum(z_img != 0).item() + torch.sum(z_text != 0).item()
        l1 += torch.sum(torch.abs(z_img)).item() + torch.sum(torch.abs(z_text)).item()
        
        # compute the reconstruction error
        img_l2 += F.mse_loss(img_hat, image_features, reduction='sum').item()
        text_l2 += F.mse_loss(text_hat, text_features, reduction='sum').item()

        img_r2 += r2_score(image_features, img_hat) * image_features.shape[0]
        text_r2 += r2_score(text_features, text_hat) * text_features.shape[0]
        
        # compute the dead features
        dead_mask += (z_img != 0).sum(dim=0) + (z_text != 0).sum(dim=0)

    l0 /= (n_images + n_texts)
    l1 /= (n_images + n_texts)
    l2 = (img_l2 + text_l2) / (n_images + n_texts)
    r2 = (img_r2 + text_r2) / (n_images + n_texts)
    
    dead_mask = (dead_mask == 0).float()

    dead_count = dead_mask.sum().item()
    total_count = dead_mask.numel()
    dead = dead_count / total_count

    return l0, l1, l2, r2, dead

##########
# Stability metrics
##########

@torch.no_grad()
def Monge(X, Y, a=None, b=None, return_plan=False, metric='arccos'):
    """
    X, Y : shape (N, d)
    a, b : shape (N, ). If None, uniform distribution
    return_plan : bool
        if True, return the plan (transportation matrix). In this case, it is a permutation matrix
        Note : if a and b are not None, as we use a barycentric projection to approximate the transport cost, there is no permutation matrix between X and Y. Instead, we return the transportation plan.
    metric : str
        'l2' : L2 distance
        'cosim' : cosine similarity
        'arccos' : arccosine distance
        default is 'arccos' as we will use it for normalized feature vectors, and this gives the geodesic distance on the sphere
        Note that L2 in this case is : sqrt(2 - 2 * cos(theta)) = sqrt(2 * cosim), since cosim is 1 - cos(theta). Arccos is simply theta
        L2 should NOT be used unless you know what you are doing ! Squared L2 should be used instead, and this is 'cosim'.
    
    Note that if a and b are not None, this function uses a barycentric projection of the Kantorovich transportation plan.
    This means that T(X) != Y, as T is a continuous approximation of the continuous Monge map between the underlying continuous distributions, discretized as X and Y.
    This means the final cost is not exactly the Monge cost, which likely does not exist in the first place.
    This also means no permutation matrix between X and Y exists.

    The reason to use non uniform distributions is to account for feature energy which might better reflect the actual geometry of the features.
    A stupid example would be if one dictionary has a very tight cluster of very low energy features. If they are not weighted down, they will dominate the distribution and severely skew the geometry.
    """

    assert metric in ['l2', 'cosim', 'arccos'], f"metric must be 'l2', 'cosim' or 'arccos'. Got {metric} instead"

    if metric == 'l2':
        C = torch.cdist(X, Y, p=2).cpu().numpy()**2 # shape (N, N)
    elif metric == 'cosim':
        C = X @ Y.T # shape (N, N)
        C = 1 - C # X and Y are already normalized
    elif metric == 'arccos':
        C = X @ Y.T
        C = torch.acos(torch.clamp(C, -1 + 1e-7, 1 - 1e-7)).cpu().numpy() # shape (N, N)
    
    if a is None and b is None:
        row_ind, col_ind = linear_sum_assignment(C) # shape (N, )
        cost = C[row_ind, col_ind].sum() # shape (1, )
        cost /= X.shape[0] # shape (1, )

        if return_plan:
            N = X.shape[0]
            P = np.zeros((N, N))
            P[row_ind, col_ind] = 1
            return cost, P
        return cost
    else:
        gamma = ot.emd(a, b, C) # shape (N, N)

        # Compute Yi = T_bary(Xi) = sum_j{gamma_ij * Yj} / sum_j{gamma_ij}
        X = X.cpu().numpy() # shape (N, d)
        Y = Y.cpu().numpy() # shape (N, d)
        Y_bary = (gamma @ Y) / gamma.sum(axis=1, keepdims=True) # shape (N, d)

        # Compute the cost between X and T(X)
        if metric == 'l2':
            diff = X - Y_bary
            cost = (diff**2).sum(axis=1) # shape (N, )
            cost = (a * cost).sum() # shape (1, )
        elif metric == 'cosim':
            X_norm = torch.linalg.norm(X, ord=2, dim=1).cpu().numpy() # shape (N, )
            X_norm = X / X_norm[:, None] # shape (N, d)
            Y_bary_norm = torch.linalg.norm(Y_bary, ord=2, dim=1) # shape (N, )
            Y_bary_norm = Y_bary / Y_bary_norm[:, None] # shape (N, d)
            cosim = (X_norm * Y_bary_norm).sum(axis=1) # shape (N, )
            cost = (a * (1 - cosim)).sum() # shape (1, )
        elif metric == 'arccos':
            X_norm = torch.linalg.norm(torch.tensor(X), ord=2, dim=1).cpu().numpy() # shape (N, )
            X_norm = X / X_norm[:, None] # shape (N, d)
            Y_bary_norm = torch.linalg.norm(torch.tensor(Y_bary), ord=2, dim=1).cpu().numpy() # shape (N, )
            Y_bary_norm = Y_bary / Y_bary_norm[:, None] # shape (N, d)
            cosim = (X_norm * Y_bary_norm).sum(axis=1) # shape (N, )
            cost = (a * torch.acos(torch.clamp(torch.tensor(cosim), -1 + 1e-7, 1 - 1e-7)).cpu().numpy()).sum() # shape (1, )

        if return_plan:
            return cost, gamma
        return cost

@torch.no_grad()
def Wasserstein(X, Y, a=None, b=None, return_plan=False, return_C=False, metric='l2'):
    assert metric in ['l2', 'cosim', 'arccos'], f"metric must be 'l2', 'cosim' or 'arccos'. Got {metric} instead"

    if metric == 'l2':
        C = torch.cdist(X, Y, p=2).cpu().numpy()**2 # shape (N, N)
    elif metric == 'cosim':
        nx = torch.linalg.norm(X, ord=2, dim=1) # shape (N, )
        ny = torch.linalg.norm(Y, ord=2, dim=1) # shape (N, )
        nx[nx.abs() < 1e-7] = 1 # avoid division by zero
        ny[ny.abs() < 1e-7] = 1
        X = X / nx[:, None] # shape (N, d)
        Y = Y / ny[:, None] # shape (N, d)
        C = X @ Y.T # shape (N, N)
        C = (1 - C).cpu().numpy()
    elif metric == 'arccos':
        C = X @ Y.T
        C = torch.acos(torch.clamp(C, -1 + 1e-7, 1 - 1e-7)).cpu().numpy() # shape (N, N)
    if a is None:
        a = np.ones(X.shape[0]) / X.shape[0]
    if b is None:
        b = np.ones(Y.shape[0]) / Y.shape[0]

    gamma = ot.emd(a, b, C) # shape (N, N)
    cost = (gamma * C).sum() # shape (1, )

    if return_plan:
        if return_C:
            return cost, gamma, C
        return cost, gamma
    if return_C:
        return cost, C
    return cost
    

@torch.no_grad()
def Sinkhorn(X, Y, a=None, b=None, return_plan=False):
    ...

##########
# D structure / geometry metrics
##########

def OOD_score(D, data_loader, metric='cosim'):
    # How far is each feature vector from the closest embedding? Again, mean vs weighted mean?
    # "The OOD score quantifies the distance of each feature vector from the closest embedding, providing insights into the model's generalization capabilities"
    assert metric in ['cosim', 'dist'], f"metric must be 'cosim' or 'dist'. Got {metric} instead"
    
    # 1 - 1/nb_concepts sum_i{max_j{metric(D_i, data_j)}}
    # Not sure it is pertinent to want features to be close to embeddings. Will implement it later
    ...

def stable_rank(D, w=None):
    # stable
    # smooth proxy for rank estimation in high dimensions - numerical errors : all dictionaries are nearly full rank
    # "The stable rank provides a smooth proxy for rank estimation in high dimensions, addressing numerical errors that can arise when dictionaries are nearly full rank"
    
    if w is not None:
        # D = D * w (shape (nb_concepts, d) * shape (nb_concepts, 1))
        D = D * w
    return (torch.linalg.norm(D, ord='fro') ** 2 / torch.linalg.norm(D, ord=2) ** 2).item() # shape (1, )

def effective_rank(D, w=None):
    # effective
    # entropy of the singular value distribution
    # "An effective rank close to k suggests that all dictionary atoms are equally important, whereas a low effective rank implies that only a few dominant concepts capture most of the variation in the data"
    
    if w is not None:
        # D = D * w (shape (nb_concepts, d) * shape (nb_concepts, 1))
        D = D * w

    U, S, V = torch.linalg.svd(D)
    S = S / S.sum()
    S = S[S > 0] # remove zero singular values

    return torch.exp(-torch.sum(S * torch.log(S))).item() # shape (1, )

def coherence(D):
    # "quantifies redundancy between dictionary atoms by measuring the maximum pairwise cosine similarity"
    
    # D shape (nb_concepts, D)
    DD = D @ D.T # shape (nb_concepts, nb_concepts)
    DD = DD - torch.eye(DD.shape[0], device=DD.device) * DD.diagonal() # remove diagonal

    return DD.max().item()

##########
# Z structure
##########

@torch.no_grad()
def stable_rank_Z(sae, data_loader, w=None):
    # stable
    # smooth proxy for rank estimation in high dimensions - numerical errors : all dictionaries are nearly full rank
    # "The stable rank provides a smooth proxy for rank estimation in high dimensions, addressing numerical errors that can arise when dictionaries are nearly full rank"
    
    Z = 0
    n_images = 0
    n_texts = 0

    for image_features, text_features in tqdm(data_loader):
        n_images += image_features.shape[0]
        n_texts += text_features.shape[0]

        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, _ = sae(image_features) # shape (N, nb_concepts)
        _, z_text, _ = sae(text_features)
        Z += (z_img.T @ z_img) + (z_text.T @ z_text)
    Z /= (n_images + n_texts)
    if w is not None:
        # Z = Z * w (shape (nb_concepts, nb_concepts) * shape (nb_concepts, 1))
        Z = Z * w
        
    return (torch.linalg.norm(Z, ord='fro') ** 2 / torch.linalg.norm(Z, ord=2) ** 2).item() # shape (1, )

@torch.no_grad()
def connectivity(sae, data_loader):
    # "measures the diversity of concept usage by counting the number of unique co-activations within the code matrix"
    
    # 1 - (1/d^2) ||Z^TZ||_0 d :
    # Z : shape (N, nb_concepts), Z^TZ : shape (nb_concepts, nb_concepts), ||...||_0 : how many unique pairs of co-activations

    Z = 0
    for image_features, text_features in tqdm(data_loader):
        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, _ = sae(image_features) # shape (N, nb_concepts)
        _, z_text, _ = sae(text_features)

        Z += (z_img.T @ z_img) + (z_text.T @ z_text)

    return 1 - (Z != 0).sum() / (Z.shape[0] ** 2)

@torch.no_grad()
def negative_interference(sae, data_loader, D):
    # "quantifies the extent to which co-activated concepts cancel each other out, reducing their effectiveness"
    # ||ReLU(- (Z^T Z) * (D^T D))||_2
    Z = 0
    for image_features, text_features in tqdm(data_loader):
        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, _ = sae(image_features) # shape (N, nb_concepts)
        _, z_text, _ = sae(text_features)

        Z += (z_img.T @ z_img) + (z_text.T @ z_text)

    D = D @ D.T # shape (nb_concepts, nb_concepts)

    return torch.linalg.norm(F.relu(-Z * D), ord=2).item() # shape (1, )

@torch.no_grad()
def stable_effective_connectivity_neginter(sae, data_loader, D):
    Z = 0
    n_images = 0
    n_texts = 0
    for image_features, text_features in tqdm(data_loader):
        n_images += image_features.shape[0]
        n_texts += text_features.shape[0]
        
        image_features = image_features.to(sae.device)
        text_features = text_features.to(sae.device)

        # get the codes
        _, z_img, _ = sae(image_features) # shape (N, nb_concepts)
        _, z_text, _ = sae(text_features)
        
        # normalize z_img and z_text st z_img[i] has norm 1
        z_img = z_img / torch.linalg.norm(z_img, ord=2, dim=1, keepdim=True)
        z_text = z_text / torch.linalg.norm(z_text, ord=2, dim=1, keepdim=True)

        Z += (z_img.T @ z_img) + (z_text.T @ z_text)
    Z /= (n_images + n_texts)
    
    # connectivity
    conn = 1 - (Z != 0).sum() / (Z.shape[0] ** 2)
    
    # negative interference
    DDT = D @ D.T
    DDT.mul_(Z)
    DDT.mul_(-1)
    DDT = F.relu(DDT, inplace=True)
    eigvals = torch.linalg.eigvalsh(DDT.cpu())
    neginter = eigvals.abs().max().item()
    
    # rank calculations
    eigvals = torch.linalg.eigvalsh(Z.cpu())
    normfro = torch.linalg.norm(Z.cpu(), ord='fro').item() ** 2
    norm2 = eigvals.abs().max().item() ** 2
    stable = (normfro / norm2)

    effective = torch.exp(-torch.sum(eigvals[eigvals > 0] * torch.log(eigvals[eigvals > 0]))).item()
    
    return stable, effective, conn, neginter

##########
# Feature linearity metrics
##########

def feature_linearity(sae, data_loader):
    ...