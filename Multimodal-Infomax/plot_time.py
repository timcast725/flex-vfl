"""
Plot experiment comparing Flex-VFL, Sync-VFL, and Async-VFL
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import random
import os
import glob

font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
        'size'   : 20}
plt.rc('font', **font)
colors=['#6aa2fc', '#fc8181', '#a5ff9e', '#3639ff', '#ff3636', '#13ba00', '#ff62f3']

def smooth(arr, std):
    first_val = -1
    first_std = -1
    val_ind = -1
    for i in range(len(arr)):
        arr[i] = np.max(arr[:i+1])
    for i in range(len(arr)):
        if arr[i] != first_val:
            if val_ind != -1:
                smoothed_vals = np.arange(first_val, arr[i], (arr[i]-first_val)/(i-val_ind))
                smoothed_stds = np.arange(first_std, std[i], (std[i]-first_std)/(i-val_ind))
                if len(smoothed_vals) != len(arr[val_ind:i]):
                    smoothed_vals = smoothed_vals[:len(smoothed_vals)-1]
                if len(smoothed_stds) != len(std[val_ind:i]):
                    smoothed_stds = smoothed_stds[:len(smoothed_stds)-1]
                arr[val_ind:i] = smoothed_vals
                std[val_ind:i] = smoothed_stds 
            first_val = arr[i]
            first_std = std[i]
            val_ind = i
    return np.array(arr), np.array(std)

def all_seeds(prefix, time_per_epoch):
    print(prefix)
    files = glob.glob(f'{prefix}_lrmain0.001_lrbert5e-05_decay0.0001_seed*.pkl')
    pickles = []    
    min_len = 99999
    for f in files:
        pkl = pickle.load(open(f, 'rb'))
        pkl = np.array([float(val['mae']) for val in pkl])
        if 'pbcd' in prefix and len(pkl) <= 10:
            continue
        if '50' in prefix and len(pkl) <= 1000:
            continue
        pickles.append(pkl)
        min_len = min(min_len, len(pkl))
    for i in range(len(pickles)):
        pickles[i] = pickles[i][:min_len]
    pickles = np.array(pickles)

    time_to_reach = []
    for i in range(len(pickles)):
        final_epoch = 0
        time_so_far = 0
        while pickles[i][final_epoch] > 0.65:
            time_so_far += time_per_epoch
            final_epoch += 1
            if final_epoch >= len(pickles[i]):
                time_so_far = float('inf')
                break
        time_to_reach.append(time_so_far/1000)
    time_2 = []
    for i in range(len(time_to_reach)):
        if time_to_reach[i] != float('inf'):
            time_2.append(time_to_reach[i])
    if len(time_2) > 0:
        time_to_reach = time_2
    time_to_reach = np.array(time_to_reach)
    min_ind, max_ind = np.argmin(time_to_reach), np.argmax(time_to_reach)
    #time_to_reach = np.delete(time_to_reach, [min_ind, max_ind])
    time_avg = np.average(time_to_reach)
    time_std = np.std(time_to_reach)
    print(f'& {time_avg:.2f} $\pm$ {time_std:.2f}')
    
    #pickles = np.delete(pickles, [min_ind, max_ind], axis=0)
    avg = np.average(pickles, axis=0)
    #print(avg)
    std = np.std(pickles, axis=0)

    return (avg, std)

num_batches = 326
epochs = 2000
local_time = 1
Q = 20
slowest = 20/5

# For each amount of server round-trip communication
comm_times = [1.0, 10.0, 50.0]
for comm_time in comm_times:
    # Parse results
    accs3, std3 = all_seeds(f"results_algpbcd_servertime_1.0_datasetmosei", (comm_time+local_time*slowest)*num_batches)
    accs2, std2 = all_seeds(f"results_algvafl_servertime_{comm_time}_datasetmosei", num_batches)
    accs4, std4 = all_seeds(f"results_algsync1_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches)
    accs5, std5 = all_seeds(f"results_algsync_servertime_1.0_datasetmosei", (comm_time+local_time*Q*slowest)*num_batches)
    accs1, std1 = all_seeds(f"results_algflex_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches) 
    #accs3, std3 = all_seeds(f"results_videotext_algpbcd_servertime_1.0_datasetmosei", (comm_time+local_time*slowest)*num_batches)
    #accs2, std2 = all_seeds(f"results_videotext_algvafl_servertime_{comm_time}_datasetmosei", num_batches)
    #accs4, std4 = all_seeds(f"results_videotext_algsync_servertime_1.0_datasetmosei", (comm_time+local_time*Q*slowest)*num_batches)
    #accs1, std1 = all_seeds(f"results_videotext_algflex_servertime_1.0_datasetmosei", (comm_time+local_time*Q)*num_batches) 
    #lrs1 = ['0.005','0.001']
    #lrs2 = ['5e-05']
    #decays = ['0.001','0.0001','1e-05','0.0']
    #for lr1 in lrs1:
    #    for lr2 in lrs2:
    #        for decay in decays:
    #            accs1, std1 = all_seeds(f"results_algflex_servertime_1.0_datasetmosei_lrmain{lr1}_lrbert{lr2}_decay{decay}", (comm_time+local_time*Q)*num_batches) 
    #            print(accs1)
    #            max_smoothness = 0
    #            prev_acc = 0
    #            smoothness = 0
    #            for acc in accs1:
    #                if acc <= prev_acc:
    #                    smoothness += 1
    #                else:
    #                    max_smoothness = max(smoothness, max_smoothness)
    #                    smoothness = 0
    #                prev_acc = acc
    #            print("smoothness", max_smoothness)


    ## Parse VAFL for the given server comm time

    ## Convert x-axis to time in milliseconds
    #VAFL_time = num_batches 
    ## Flex-VFL 
    #Our_ratio = VAFL_time/((comm_time+local_time*Q)*num_batches)
    ## Sync-VFL
    #Sync_ratio = VAFL_time/((comm_time+local_time*Q*slowest)*num_batches)
    ## P-BCD
    #Q1_ratio = VAFL_time/((comm_time+local_time*slowest)*num_batches)

    ## Stretch Flex-VFL results to be on same time scale as VAFL
    #r = 0
    #j = 0
    #accs1new = []
    #std1new = []
    #for i in range(len(accs2)):
    #    if j >= len(accs1):
    #        break
    #    accs1new.append(float(accs1[j]))
    #    std1new.append(float(std1[j]))
    #    r += Our_ratio 
    #    if r >= 1:
    #        r -= 1
    #        j += 1

    ## Stretch P-BCD results to be on same time scale as VAFL
    #r = 0
    #j = 0
    #accs3new = []
    #std3new = []
    #for i in range(len(accs2)):
    #    if j >= len(accs3):
    #        break
    #    accs3new.append(float(accs3[j]))
    #    std3new.append(float(std3[j]))
    #    r += Q1_ratio 
    #    if r >= 1:
    #        r -= 1
    #        j += 1

    ## Stretch Sync-VFL results to be on same time scale as VAFL
    #r = 0
    #j = 0
    #accs4new = []
    #std4new = []
    #for i in range(len(accs2)):
    #    if j >= len(accs4):
    #        break
    #    accs4new.append(float(accs4[j]))
    #    std4new.append(float(std4[j]))
    #    r += Sync_ratio 
    #    if r >= 1:
    #        r -= 1
    #        j += 1

    ## Plot loss
    #time = np.arange(0, (epochs+2)*num_batches, num_batches)
    ## Plot accuracy
    #fig, ax = plt.subplots()

    #accs2 = np.array(accs2, dtype=float)
    #std2 = np.array(std2, dtype=float)
    #accs1new, std1new = smooth(accs1new, std1new)
    #accs2new, std2new = smooth(accs2, std2)
    #accs3new, std3new = smooth(accs3new, std3new)
    #accs4new, std4new = smooth(accs4new, std4new)
    #plt.plot(time[:len(accs1new)], accs1new, color=colors[3], label='Flex-VFL')
    #plt.plot(time[:len(accs2new)], accs2new, color=colors[4], label='VAFL')
    #plt.plot(time[:len(accs4new)], accs4new, color=colors[5], label='Sync-VFL')
    #plt.plot(time[:len(accs3new)], accs3new, color=colors[6], label='P-BCD')

    #plt.fill_between(np.linspace(0,time[len(accs1new)-1],len(accs1new)), accs1new - std1new, accs1new + std1new, alpha=0.3)
    #plt.fill_between(np.linspace(0,time[len(accs2new)-1],len(accs2new)), accs2new - std2new, accs2new + std2new, alpha=0.3)
    #plt.fill_between(np.linspace(0,time[len(accs4new)-1],len(accs4new)), accs4new - std4new, accs4new + std4new, alpha=0.3)
    #plt.fill_between(np.linspace(0,time[len(accs3new)-1],len(accs3new)), accs3new - std3new, accs3new + std3new, alpha=0.3)

    #plt.xlabel('Time units')
    #plt.ylabel('F1-Score')

    #plt.legend(loc='lower right')

    #plt.tight_layout()
    #plt.savefig(f'acc_Multimodal_{comm_time}.png')
