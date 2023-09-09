import logging

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger()


def dirichlet_kl_divergence(z:torch.Tensor, alpha_q:torch.Tensor, alpha_p: torch.Tensor, q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution):


#     S_alpha = torch.sum(alpha_q, dim=1, keepdim=True)
#     S_beta = torch.sum(alpha_p, dim=1, keepdim=True)
#     lnB = torch.lgamma(S_alpha) - \
#         torch.sum(torch.lgamma(alpha_q), dim=1, keepdim=True)
#     lnB_uni = torch.sum(torch.lgamma(alpha_p), dim=1,
#                         keepdim=True) - torch.lgamma(S_beta)

#     dg0 = torch.digamma(S_alpha)
#     dg1 = torch.digamma(alpha_q)

#     kl = torch.sum((alpha_q - alpha_p) * (dg1 - dg0), dim=1,
#                    keepdim=True) + lnB + lnB_uni
    
    kl = F.kl_div(q_distrib.rsample(), p_distrib.rsample())
    q_logprob = q_distrib.log_prob(z)
    p_logprob = p_distrib.log_prob(z)
    
    return kl,  q_logprob, p_logprob


# def dirichlet_kl_divergence(z, alpha_q:torch.Tensor, alpha_p: torch.Tensor, q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution):
#     """
#     Estimate the KL divergence between two Dirichlet distributions using analytical approach.
    
#     Parameters:
#         alpha_p (torch.Tensor): Parameters of the first Dirichlet distribution.
#         alpha_q (torch.Tensor): Parameters of the second Dirichlet distribution.
    
#     Returns:
#         torch.Tensor: KL divergence between the two Dirichlet distributions.
#     """
#     # Calculate the KL divergence term for each dimension
#     kl_elementwise = F.kl_div(q_distrib.rsample().log(), p_distrib.rsample().log(), log_target=True, reduction='none')

#     q_logprob = q_distrib.log_prob(z)
#     p_logprob = p_distrib.log_prob(z)
    
#     return kl_elementwise, q_logprob, p_logprob

def loglikelihood_loss(z: torch.Tensor, alpha: torch.Tensor):
    """Calculate log likelihood of Dir sample"""
    # z, alpha = z.float(), alpha.float()
    s = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (z - (alpha / s)) ** 2, dim=1, keepdim=True)
    
    loglikelihood_var = torch.sum(
        alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)
    
    loglikelihood = loglikelihood_err + loglikelihood_var
    
    return -loglikelihood


# def dirichlet_log_likelihood(x, alpha):
#     # Ensure x and alpha have the same length
#     assert len(x) == len(alpha), "x and alpha must have the same length"
    
#     # Compute the logarithm of the multivariate Beta function
#     log_beta = torch.sum(gammaln(alpha), keepdim=True) - gammaln(torch.sum(alpha, keepdim=True))
    
#     # Compute the sum of terms for the log likelihood
#     log_likelihood = log_beta + torch.sum((alpha - 1) * np.log(x), keepdim=True)
#     return log_likelihood

def kl_divergence_mc(
    z: torch.Tensor, q_alpha: torch.Tensor, p_alpha: torch.Tensor
):
    """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

    Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

    Args:
        z: Samples
        q_distrib: First distribution (Variational distribution)
        p_distrib: Second distribution

    Returns:
        tuple: Spatial KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
    """

    q_logprob = loglikelihood_loss(z, q_alpha)
    p_logprob = loglikelihood_loss(z, p_alpha)
    
    kl_elementwise = q_logprob - p_logprob
    
    if kl_elementwise.isnan().any():
        LOGGER.warning(f"Encountered `nan` in KL divergence of shape {kl_elementwise.shape=}.")
        
    return kl_elementwise, q_logprob, p_logprob

# def kl_divergence_mc(
#     z: torch.Tensor, q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution, multinomial=False
# ):
#     """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

#     Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

#     Args:
#         z: Samples
#         q_distrib: First distribution (Variational distribution)
#         p_distrib: Second distribution

#     Returns:
#         tuple: Spatial KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
#     """
#     q_logprob = q_distrib.log_prob(z)
#     p_logprob = p_distrib.log_prob(z)
    
#     kl_elementwise = q_logprob - p_logprob
    
#     if multinomial:
#         kl_elementwise = torch.tile(kl_elementwise, (z.shape[0], 1))
#     if kl_elementwise.isnan().any():
#         LOGGER.warning(f"Encountered `nan` in KL divergence of shape {kl_elementwise.shape=}.")
        
        
#     return kl_elementwise, q_logprob, p_logprob

