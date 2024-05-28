from GaussianMixDataGenerator.data.utils import AUCFromDistributions
from GaussianMixDataGenerator.data.datagen import MVNormalMixDG as GMM
import numpy as np




def buildGaussianMixDataGeneratorDim(dim, alpha):
    mu_pos = np.random.random_sample(size = dim)
    mu_neg = mu_pos - np.random.random_sample(size = dim)*0.3
    cov_pos = np.diag(np.full(dim,1))
    cov_neg = np.diag(np.full(dim,1))
    
    return GMM([mu_pos], [cov_pos], [1], [mu_neg], [cov_neg], [1], alpha)

def buildGaussianMixDataGenerator(mu_pos=None, mu_neg=None,sig_pos=None,sig_neg=None):

    alpha = None
    p_pos = [1.0]
    p_neg = [1.0]
    if mu_pos is None:
        mupos = [0.5]
        muneg = [-0.5]
        sigpos = [1.0]
        signeg = [1.0]
        return GMM(mupos, sigpos, p_pos, muneg, signeg, p_neg, alpha)
    else:
        print(mu_pos, mu_neg, sig_pos, sig_neg, "  here")
        assert (mu_pos is not None and mu_neg is not None and sig_pos is not None and sig_neg is not None)
        return GMM([mu_pos], [sig_pos], p_pos, [mu_neg], [sig_neg], p_neg, [alpha])

