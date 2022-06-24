from torch.distributions.multivariate_normal import MultivariateNormal
import torch


def vae_loss(x, x_recons, log_p_zt, l, expected=True, beta = 1):
    
    recons = (x-x_recons).pow(2).sum(axis = 1)
    KLD = -log_p_zt+l

    if expected:
        ''' Used in training '''
        # print(f"Ratio: {recons.sum(axis = 0)/torch.mean(KLD)}, recons: {recons.sum(axis = 0)}, KLD: {torch.mean(KLD)}")
        return recons.sum(axis = 0) + beta*torch.mean(KLD)


    recons = torch.nan_to_num(recons, nan=500)
    recons = torch.minimum(recons, torch.tensor(1000)) 

    KLD = torch.nan_to_num(KLD, nan=10000)
    KLD = torch.minimum(KLD, torch.tensor(10000))
    beta = KLD.mean()/ recons.mean()
    KLD = KLD/beta
    
    '''
    print(f"Recons: max {recons.max()}, min: {recons.min()}, mean: {recons.mean()}")
    print(f"KLD: max {KLD.max()}, min: {KLD.min()}, mean: {KLD.mean()}")
    print(recons, '\n', KLD)
    '''

    return recons + KLD




