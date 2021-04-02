"""
Testing for 'knapsack' distribution between two vectors
"""

import numpy as np
from itertools import product

# Given: list of [GPU, CPU] weights
# Assume CPU time > GPU time
def simulate(n_weights=10):
    wgts = [sorted([np.random.rand(), np.random.rand()]) for i in range(n_weights)]

    # Goal: select one from each col, minimize difference between the two sums
    gpus, cpus = [], []
    wgts = sorted(wgts, key=lambda wgt:wgt[1])
    for wgt in wgts:
        if sum(gpus) < sum(cpus):
            gpus.append(wgt[0])
        else:
            cpus.append(wgt[1])

    efficiency = lambda a, b: np.mean([sum(a), sum(b)])/max([sum(a), sum(b)])
    eff = efficiency(gpus, cpus)
    config = [gpus, cpus]

    # Now compute the optimal
    args = [[1, 0] for _ in range(len(wgts))]
    eff_max, best_selection = -float('inf'), []
    for p in product(*args):
        gpus, cpus = [], []
        for i in range(len(wgts)):
            if p[i] == 0:
                gpus.append(wgts[i][0])
            else:
                cpus.append(wgts[i][1])

        eff_max = max(eff_max, efficiency(cpus, gpus))
        if eff_max == efficiency(gpus, cpus):
            best_config = [gpus, cpus]


    # Print useful info -- configs are config, best_config
    print("eff (greedy ): {:8.7f}, eff (optimal): {:8.7f}".format(eff, eff_max))


    return eff, eff_max


n_trials = 100
r = []
for _ in range(n_trials):
    eff, eff_max = simulate(10)
    r.append(eff/eff_max)

print("Algorithm gets to {}(%) of best possible in {} trials.".format(
    np.mean(r), n_trials))

