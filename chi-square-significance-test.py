import scipy.stats as ss
import numpy as np
import scikit_posthocs as sp
from scipy.stats import wilcoxon

data = np.array([
    # Samples (datasets) ->                                    ↓ Treatments
    [14.4540, 17.3168, 14.7477, 38.3733, 4.9597, 0.3335, 3.6756, 23.4954],  # FastGCN
    [0.7089, 4.0147, 18.6169, 0.1393, 0.3329, 0.9420, 0.7474, 0.1607],  # LADIES
    [0.0016, 0.0130, 0.0013, 0.1304, 0.1004, 0.0232, 0.3967, 0.2358],  # GraphSAINT
    [0.0602, 0.0137, 0.0033, 0.2550, 0.6420, 9.2825, 0.0599, -1.00],  # AS-GCN (OOM replaced with -1)
    [0.0047, 0.0616, 0.0018, 0.0209, 0.0911, 0.0084, 0.0085, 0.0002],  # GRAPES
])

no_asgcn = np.concatenate((data[:4], data[5:6]))
print(f'Means without AS-GCN={np.mean(no_asgcn, axis=1)}')
print(f'AS-GCN mean={np.mean(data[4, :7])}')

print(ss.friedmanchisquare(*data))
print(sp.posthoc_nemenyi_friedman(data.T) <= 0.05)

ranks = np.argsort(np.argsort(data, axis=0), axis=0) + 1
print(np.sum(ranks, axis=1))
print(f'mean_ranks={np.mean(ranks, axis=1)}\n')

print(f'Wilcoxon signed-rank test with Bonferroni correction:')
models = ['FastGCN', 'LADIES', 'GraphSAINT', 'AS-GCN', 'GRAPES']
gfgs_ranks = ranks[-1]
num_comparisons = len(models) - 1
for i in range(len(models) - 1):
    model_ranks = ranks[i]
    stat, p = wilcoxon(gfgs_ranks, model_ranks)
    p = p * num_comparisons
    print(f'{models[i]}: stat={stat}, p={p}' + ('***' if p < 0.05 else ''))
