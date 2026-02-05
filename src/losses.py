"""
Module containing loss functions for the Sparse Autoencoder (SAE) model.

In the Overcomplete library, the loss functions are defined as standalone functions.
They all share the same signature:
    - x: torch.Tensor
        Input tensor.
    - x_hat: torch.Tensor
        Reconstructed tensor.
    - pre_codes: torch.Tensor
        Encoded tensor before activation function.
    - codes: torch.Tensor
        Encoded tensor.
    - dictionary: torch.Tensor
        Dictionary tensor.

Additional arguments can be passed as keyword arguments.
"""

from functools import partial

import torch
from overcomplete.metrics import hoyer, kappa_4, lp, l1, dead_codes

# disable W0613 (unused-argument) to keep the same signature for all loss functions
# pylint: disable=W0613


def _mse_with_penalty(x, x_hat, pre_codes, codes, dictionary, penalty=1.0, penalty_fn=None, **kwargs):
    """
    Compute the Mean Squared Error (MSE) loss with a given penalty function.

    Loss = ||x - x_hat||^2 + penalty * penalty_fn(z)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float
        Penalty coefficient.
    penalty_fn : callable
        Penalty function.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    if isinstance(x, (list, tuple)):
        x = x[0]
    if isinstance(x_hat, (list, tuple)):
        x_hat = x_hat[0]
    if isinstance(codes, (list, tuple)):
        codes = codes[0]
    mse = (x - x_hat).square().mean()

    if penalty_fn is None:
        reg = 0.0
    else:
        reg = torch.mean(penalty_fn(codes)) * penalty

    return mse + reg


# l1 should be dependent on the codes dimension, hoyer and kappa are not
mse_l1 = partial(_mse_with_penalty, penalty_fn=l1)
mse_hoyer = partial(_mse_with_penalty, penalty_fn=hoyer)
mse_kappa_4 = partial(_mse_with_penalty, penalty_fn=kappa_4)


def mse_elastic(x, x_hat, pre_codes, codes, dictionary, alpha=0.5):
    """
    Compute the Mean Squared Error (MSE) loss with L1 penalty on the codes.

    Loss = ||x - x_hat||^2 + (1 - alpha) * ||z||_1 + alpha * ||D||^2

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    alpha : float, optional
        Alpha coefficient in the Elastic-net loss, control the ratio of l1 vs l2.
        alpha=0 means l1 only, alpha=1 means l2 only.

    Returns
    -------
    torch.Tensor
        Loss value.
    """
    assert 0.0 <= alpha <= 1.0

    mse = (x - x_hat).square().mean()

    l1_loss = codes.abs().mean()
    l2_loss = dictionary.square().mean()

    loss = mse + (1.0 - alpha) * l1_loss + alpha * l2_loss

    return loss


def top_k_auxiliary_loss(x, x_hat, pre_codes, codes, dictionary, penalty=0.1):
    """
    The Top-K Auxiliary loss (AuxK).

    The loss is defined in the original Top-K SAE paper:
        "Scaling and evaluating sparse autoencoders"
        by Gao et al. (2024).

    Similar to Ghost-grads, it consist in trying to "revive" the dead codes
    by trying the predict the residual using the 50% of the top non choosen codes.

    Loss = ||x - x_hat||^2 + penalty * ||x - (x_hat D * top_half(z_pre - z)||^2

    @tfel the order actually matter here! residual is x - x_hat and
    should be in this specific order.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    """
    # select the 50% of non choosen codes and predict the residual
    # using those non choosen codes
    # the code choosen are the non-zero element of codes

    residual = x - x_hat
    mse = residual.square().mean()

    pre_codes = torch.relu(pre_codes)
    pre_codes = pre_codes - codes  # removing the choosen codes

    auxiliary_topk = torch.topk(pre_codes, k=pre_codes.shape[1] // 2, dim=1)
    pre_codes = torch.zeros_like(codes).scatter(-1, auxiliary_topk.indices,
                                                auxiliary_topk.values)

    residual_hat = pre_codes @ dictionary
    auxilary_mse = (residual - residual_hat).square().mean()

    loss = mse + penalty * auxilary_mse

    return loss


def reanimation_regularizer(x, x_hat, pre_codes, codes, dictionary, penalty=0.1):
    """
    Additional term to the loss function that tries to revive dead codes.

    The idea is to increase the value of the `pre_codes` to allow the dead_codes
    to fire again by pushing the `pre_codes` towards the positive orthant.

        reg = -1/(sum_i dead_mask_i) * sum_i(pre_codes_i * dead_mask_i).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    """
    # codes that havent fired in this batch
    dead_mask = dead_codes(codes)

    # push the `pre_codes` towards the positive orthant
    reg = -(pre_codes * dead_mask).sum() / (dead_mask.sum() + 1e-6)
    reg = reg * penalty

    return reg


def alignment_penalty(x, x_hat, pre_codes, codes, dictionary, penalty=0.03, alignment_metric="cosim"):
    """
    Additional term to the loss function that tries to the first and second half of the codes.
    In multimodal cases (or when joint training a pair of dictionaries),
    one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    x_hat : torch.Tensor
        Reconstructed tensor.
    pre_codes : torch.Tensor
        Encoded tensor before activation function.
    codes : torch.Tensor
        Encoded tensor.
    dictionary : torch.Tensor
        Dictionary tensor.
    penalty : float, optional
        Penalty coefficient.
    alignment_metric : str, optional
        Metric to use for the alignment penalty. Can be "l1" or "l2".
        Default is "l1".
    """

    # split the codes into two halves
    assert codes.shape[0] % 2 == 0, "batch size must be even"
    n = codes.shape[0] // 2
    
    # codes_1 = codes[:n] # shape: (n, d)
    # codes_2 = codes[n:] # shape: (n, d)

    # arange = torch.arange(n, device=codes.device)

    # mask_1 = torch.ones_like(codes_1)
    # mask_2 = torch.ones_like(codes_2)
    
    # k = 2
    # top_k_idxs_1 = torch.topk(codes_1, k=k, dim=1).indices # shape: (n, k)
    # top_k_idxs_2 = torch.topk(codes_2, k=k, dim=1).indices # shape: (n, k)
    # for i in range(k):
    #     top_i_idxs_1 = top_k_idxs_1[:, i]  # shape: (n,)
    #     top_i_idxs_2 = top_k_idxs_2[:, i]  # shape: (n,)
    #     mask_1[arange, top_i_idxs_1] = 0.0
    #     mask_1[arange, top_i_idxs_2] = 0.0
    #     mask_2[arange, top_i_idxs_1] = 0.0
    #     mask_2[arange, top_i_idxs_2] = 0.0
    
    codes_1 = codes[:n] # shape: (n, d)
    codes_2 = codes[n:] # shape: (n, d)

    top_k_idxs_1 = torch.topk(codes_1, k=1, dim=1).indices.squeeze(1) # shape: (n,)
    top_k_idxs_2 = torch.topk(codes_2, k=1, dim=1).indices.squeeze(1) # shape: (n,)

    arange = torch.arange(n, device=codes.device)

    mask_1 = torch.ones_like(codes_1)
    mask_2 = torch.ones_like(codes_2)
    mask_1[arange, top_k_idxs_1] = 0.0
    mask_1[arange, top_k_idxs_2] = 0.0
    mask_2[arange, top_k_idxs_1] = 0.0
    mask_2[arange, top_k_idxs_2] = 0.0

    masked_1 = codes_1 * mask_1
    masked_2 = codes_2 * mask_2

    # compute the alignment penalty
    if alignment_metric == "l1":
        misalignment_score = (masked_1 - masked_2).abs().mean()
    elif alignment_metric == "l2":
        misalignment_score = (masked_1 - masked_2).square().mean()
    elif alignment_metric == "cosim" or "cosine" or "cosine_similarity":
        misalignment_score = - torch.nn.functional.cosine_similarity(masked_1, masked_2, dim=1).mean()
    else:
        raise ValueError(f"Alignment metric must be 'l1' or 'l2', got {alignment_metric}")
    
    # print(f"Alignment penalty: {misalignment_score.item()}")
    
    return misalignment_score * penalty


def tr_loss(x, x_hat, pre_codes, codes, dictionary, f_E1=None, f_E2=None, f_D1=None, f_D2=None, *args, **kwargs):
    # translation loss : D2 o E1 (x1) = x2 and D1 o E2 (x2) = x1
    # Note on scale : there can be massive scale differences between x1 and x2, so we first normalize the targets and predictions
    #                 This problem disappears once the norm of the atoms have converged to accomodate for similar activation energy
    #                 as per IsoE and contrastive losses. But before that happens, massive scale differences dominate the loss.
    if isinstance(codes, tuple):
        codes_1, codes_2 = codes
        x_1, x_2 = x
        n = codes_1.shape[0]
    elif isinstance(codes, torch.Tensor):
        assert codes.shape[0] % 2 == 0, "batch size must be even"
        n = codes.shape[0] // 2
        codes_1 = codes[:n] # shape: (n, d)
        codes_2 = codes[n:] # shape: (n, d)
        x_1, x_2 = x[:n], x[n:]
    else:
        raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
    
    x_1_hat = f_D1(codes_2)
    x_1, x_1_hat = x_1 / (torch.norm(x_1, dim=1, keepdim=True) + 1e-8), x_1_hat / (torch.norm(x_1_hat, dim=1, keepdim=True) + 1e-8)
    l1 = (x_1 - x_1_hat).square().mean() / (x_1.square().mean() + 1e-8)
    
    x_2_hat = f_D2(codes_1)
    x_2, x_2_hat = x_2 / (torch.norm(x_2, dim=1, keepdim=True) + 1e-8), x_2_hat / (torch.norm(x_2_hat, dim=1, keepdim=True) + 1e-8)
    l2 = (x_2 - x_2_hat).square().mean() / (x_2.square().mean() + 1e-8)
    
    return l1 + l2


def cont_loss(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
    # contrastive loss : E1 (x1) = E2 (x2)
    # Note on scale : in the codes we don't want scale differences, so we don't normalize the codes.
    #                 This implies that dictionary atoms can have non unit norm ! Otherwise to reconstruct something small,
    #                 the activation energy needs to be small and can't match the activation energy of something large.
    # Note : MSE is dominated by large values, where the normalization here helps to enforce the same penalty on low frequency
    #        components. It is however quite sensitive to noise on low energy components. Same for the IsoE loss below.
    if isinstance(codes, tuple):
        codes_1, codes_2 = codes
        x_1, x_2 = x
        n = codes_1.shape[0]
    elif isinstance(codes, torch.Tensor):
        assert codes.shape[0] % 2 == 0, "batch size must be even"
        n = codes.shape[0] // 2
        codes_1 = codes[:n] # shape: (n, d)
        codes_2 = codes[n:] # shape: (n, d)
        x_1, x_2 = x[:n], x[n:]
    else:
        raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
    # l = (codes_1 - codes_2).square().mean() # Some components dominate the loss (the high frequency ones) whereas
    #                                         # we would like all components to receive similar signal. To do so,
    #                                         # we normalize by the energy of each component.
    # This behaves weird. I don't understand why.
    # I'm pretty sure I messed up my normalisation, I should normalise the codes first then compute the mse. But that doesn't work, I want the magnitudes to match... Fuck.
    # shape analysis: ((n, d) - (n, d))^2, (n, d)^2 -> mean over n -> (d,), then broadcasted division (n, d) / (d,) -> mean over n,d -> scalar
    l = (((codes_1 - codes_2).square()) / (codes_1.square().mean(dim=0) + codes_2.square().mean(dim=0) + 1e-5)).mean() 
    
    # ... # TODO : wasserstein. Same issue as in IsoE_loss_with_distribution_matching. Disentangle frequency matching and activation energy matching.
    
    return l

def _IsoE_loss_with_mean_matching(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
    # typical strength that works : 1e-1 - not so sensitive
    # IsoEnergy loss : E [E1(x1)^2] = E [E2(x2)^2].
    # Note on scale : same as cont_loss above : codes should have comparable activation energy.
    # Note : MSE is dominated by large values, where the normalization here helps to enforce the same penalty on low frequency
    #        components. It is however quite sensitive to noise on low energy components. Same for the contrastive loss above.
    if isinstance(codes, tuple):
        codes_1, codes_2 = codes
        n = codes_1.shape[0]
    elif isinstance(codes, torch.Tensor):
        assert codes.shape[0] % 2 == 0, "batch size must be even"
        n = codes.shape[0] // 2
        codes_1 = codes[:n] # shape: (n, d)
        codes_2 = codes[n:] # shape: (n, d)
    else:
        raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
    E_1 = (codes_1 ** 2).mean(dim=0) # mean of squares for each code. shape: (d,)
    E_2 = (codes_2 ** 2).mean(dim=0)
    
    misalignment_score = ((E_1 - E_2).abs() / (E_1 + E_2 + 1e-5)).mean()
    
    return misalignment_score

# TODO : in practice, high frequency components also have high activation, and low frequency components low activation.
#        This gives disproportionate distance to high frequency components.
#        To solve the issue, separate two aspects : frequency matching and activation energy matching (distribution of X | X != 0 and Y | Y != 0).
#        Assume X ~ Z_X * A_x
#        - Z_X ~ B(p_X) (Bernoulli)
#        - A_X | Z_X = 1 ~ P_X (some distribution)
#        - A_X | Z_X = 0 = 0
#        Same for Y.
#        Then, W(X, Y) = |p_X - p_Y| * E[A] under the assumption that P_X = P_Y. In practice, it's probably not too far off, and E[A_X] is monotonous with p_X.
#        Thus, low frequency components (small p_X) have disproportionately small Wasserstein distance.
#        We then separate the matching of frequency (p_X vs p_Y) and activation energy (E[A_X | Z_X = 1] vs E[A_Y | Z_Y = 1]).
#        - Frequency matching : |log(p_X / p_Y)|
#        - Activation energy matching : |E_X - E_Y| / (E_X + E_Y), where E_X = E[A_X | Z_X = 1]
def _IsoE_loss_with_distribution_matching(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs):
    # typical strength that works : ~2e-2 - very sensitive.
    if isinstance(codes, tuple):
        codes_1, codes_2 = codes
        n = codes_1.shape[0]
    elif isinstance(codes, torch.Tensor):
        assert codes.shape[0] % 2 == 0, "batch size must be even"
        n = codes.shape[0] // 2
        codes_1 = codes[:n] # shape: (n, d)
        codes_2 = codes[n:] # shape: (n, d)
    else:
        raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
    codes_1_sorted, _ = torch.sort(codes_1.abs(), dim=0)
    codes_2_sorted, _ = torch.sort(codes_2.abs(), dim=0)
    
    Ws = torch.log((codes_1_sorted - codes_2_sorted).abs().mean())
    
    return Ws

def IsoE_loss(x, x_hat, pre_codes, codes, dictionary, matching_type="mean", *args, **kwargs):
    if matching_type == "mean":
        return _IsoE_loss_with_mean_matching(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs)
    elif matching_type == "distribution":
        return _IsoE_loss_with_distribution_matching(x, x_hat, pre_codes, codes, dictionary, *args, **kwargs)

def cy_loss(x, x_hat, pre_codes, codes, dictionary, f_E1=None, f_E2=None, f_D1=None, f_D2=None, *args, **kwargs):
    # This is the full cycle loss : D2 o E1 o D1 o E2 = Id
    # Note on scale : in latent space there can be massive scale differences between the two modalities or two layers,
    #                 so we normalize the target and prediction in latent space.
    #                 This problem disappears once the norm of the atoms have converged to accomodate for similar activation energy
    #                 as per IsoE and contrastive losses. But before that happens, massive scale differences dominate the loss.
    # Note : reconstruction loss is the demi cycle loss : D1 o E1 = Id
    if isinstance(codes, tuple):
        codes_1, codes_2 = codes
        x_1, x_2 = x
        n = codes_1.shape[0]
    elif isinstance(codes, torch.Tensor):
        assert codes.shape[0] % 2 == 0, "batch size must be even"
        n = codes.shape[0] // 2
        codes_1 = codes[:n] # shape: (n, d)
        codes_2 = codes[n:] # shape: (n, d)
        x_1, x_2 = x[:n], x[n:]
    else:
        raise ValueError("codes must be either a tuple of two tensors or a single tensor")
    
    x_1_hat = f_D1(f_E2(f_D2(codes_1))[0])
    x_1, x_1_hat = x_1 / (torch.norm(x_1, dim=1, keepdim=True) + 1e-8), x_1_hat / (torch.norm(x_1_hat, dim=1, keepdim=True) + 1e-8)
    l1 = (x_1 - x_1_hat).square().mean() / (x_1.square().mean() + 1e-8)
    
    x_2_hat = f_D2(f_E1(f_D1(codes_2))[0])
    x_2, x_2_hat = x_2 / (torch.norm(x_2, dim=1, keepdim=True) + 1e-8), x_2_hat / (torch.norm(x_2_hat, dim=1, keepdim=True) + 1e-8)
    l2 = (x_2 - x_2_hat).square().mean() / (x_2.square().mean() + 1e-8)
    
    return l1 + l2

# def alignment_penalty_prime(x, x_hat, pre_codes, codes, dictionary, penalty=0.03, alignment_metric="cosim"):
#     """
#     Additional term to the loss function that tries to the first and second half of the codes.
#     In multimodal cases (or when joint training a pair of dictionaries),
#     one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor.
#     x_hat : torch.Tensor
#         Reconstructed tensor.
#     pre_codes : torch.Tensor
#         Encoded tensor before activation function.
#     codes : torch.Tensor
#         Encoded tensor.
#     dictionary : torch.Tensor
#         Dictionary tensor.
#     penalty : float, optional
#         Penalty coefficient.
#     alignment_metric : str, optional
#         Metric to use for the alignment penalty. Can be "l1" or "l2".
#         Default is "l1".
#     """

#     # split the codes into two halves
#     assert codes.shape[0] % 2 == 0, "batch size must be even"
#     n = codes.shape[0] // 2
    
#     codes_1 = codes[:n] # shape: (n, d)
#     codes_2 = codes[n:] # shape: (n, d)

#     # compute the alignment penalty
#     if alignment_metric == "l1":
#         misalignment_score = (codes_1 - codes_2).abs().mean()
#     elif alignment_metric == "l2":
#         misalignment_score = (codes_1 - codes_2).square().mean()
#     elif alignment_metric == "cosim" or "cosine" or "cosine_similarity":
#         misalignment_score = - torch.nn.functional.cosine_similarity(codes_1, codes_2, dim=1).mean()
#     else:
#         raise ValueError(f"Alignment metric must be 'l1' or 'l2', got {alignment_metric}")
    
#     # print(f"Alignment penalty: {misalignment_score.item()}")
    
#     return misalignment_score * penalty

# def alignment_penalty_distribution(x, x_hat, pre_codes, codes, dictionary, penalty=0.03):
#     """
#     Additional term to the loss function that tries to the first and second half of the codes.
#     In multimodal cases (or when joint training a pair of dictionaries),
#     one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor.
#     x_hat : torch.Tensor
#         Reconstructed tensor.
#     pre_codes : torch.Tensor
#         Encoded tensor before activation function.
#     codes : torch.Tensor
#         Encoded tensor.
#     dictionary : torch.Tensor
#         Dictionary tensor.
#     penalty : float, optional
#         Penalty coefficient.
#     alignment_metric : str, optional
#         Metric to use for the alignment penalty. Can be "l1" or "l2".
#         Default is "l1".
#     """
#     # split the codes into two halves
#     assert codes.shape[0] % 2 == 0, "batch size must be even"
#     n = codes.shape[0] // 2
    
#     codes_1 = codes[:n] # shape: (n, d)
#     codes_2 = codes[n:] # shape: (n, d)

#     # compute the alignment penalty
#     codes_1_E = (codes_1 ** 2).mean(dim=0)  # mean of squares for each code. shape: (d,)
#     codes_2_E = (codes_2 ** 2).mean(dim=0)
    
#     misalignment_score = (((codes_1_E - codes_2_E) / (codes_1_E + codes_2_E + 1e-5))).mean()
    
#     # print(f"Misalignment score: {misalignment_score.item()}, penalty: {penalty}, misalignment_score * penalty: {(misalignment_score * penalty).item()}")
    
#     return misalignment_score * penalty

# def alignment_penalty_distribution_prime(x, x_hat, pre_codes, codes, dictionary, penalty=0.03):
#     """
#     Additional term to the loss function that tries to the first and second half of the codes.
#     In multimodal cases (or when joint training a pair of dictionaries),
#     one can use this to incentivize truely multimodal features to not split into two artificially unimodal ones.

#     Parameters
#     ----------
#     x : torch.Tensor
#         Input tensor.
#     x_hat : torch.Tensor
#         Reconstructed tensor.
#     pre_codes : torch.Tensor
#         Encoded tensor before activation function.
#     codes : torch.Tensor
#         Encoded tensor.
#     dictionary : torch.Tensor
#         Dictionary tensor.
#     penalty : float, optional
#         Penalty coefficient.
#     alignment_metric : str, optional
#         Metric to use for the alignment penalty. Can be "l1" or "l2".
#         Default is "l1".
#     """

#     # split the codes into two halves
#     assert codes.shape[0] % 2 == 0, "batch size must be even"
#     n = codes.shape[0] // 2
    
#     codes_1 = codes[:n] # shape: (n, d)
#     codes_2 = codes[n:] # shape: (n, d)

#     # compute the alignment penalty
#     codes_1_E = (codes_1 ** 2).mean(dim=0)  # mean of squares for each code. shape: (d,)
#     codes_2_E = (codes_2 ** 2).mean(dim=0)
    
#     E_tot = (codes_1_E + codes_2_E) / 2
#     E_tot = E_tot.sum().item()
#     misalignment_score = (((codes_1_E - codes_2_E) / E_tot)).mean()
    
#     # print(f"Alignment penalty: {misalignment_score.item()}, penalty: {penalty}, misalignment_score * penalty: {(misalignment_score * penalty).item()}")
    
#     return misalignment_score * penalty
