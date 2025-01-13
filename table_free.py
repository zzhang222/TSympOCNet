import pickle
import numpy as np

problem = 'free'
net_type = 'TSympNet'
Cv = 0.5
k = 0.0
c = 0.0

for net_type in ['PINN', 'TSympNet']:
    for d in [16, 32, 64, 128]:
        total_time = []
        cost = []
        hmin = []
        for seed in range(1, 11):
            path = f'{problem}_{net_type}_{seed}_{d}_{Cv}_{k}_{c}'
            total_time.append(pickle.load(open(f'saved/time_{path}', "rb")))
            if net_type == 'PINN':
                cost.append(np.loadtxt(f'saved/cost_{path}.txt'))
                hmin.append(np.loadtxt(f'saved/hmin_{path}.txt'))
            else:
                cost.append(pickle.load(open(f'saved/cost_{path}', "rb")))
                hmin.append(pickle.load(open(f'saved/hmin_{path}', "rb")))
        total_time, cost, hmin = np.array(total_time), np.array(cost), np.array(hmin)
        mean_time, mean_cost, mean_hmin = np.mean(total_time), np.mean(cost), np.mean(hmin)
        std_time, std_cost, std_hmin = np.std(total_time), np.std(cost), np.std(hmin)
        idx_min_cost = np.argmin(cost)
        print(net_type, d // 2, idx_min_cost)
        print(mean_time, mean_cost, mean_hmin)
        print(std_time, std_cost, std_hmin)