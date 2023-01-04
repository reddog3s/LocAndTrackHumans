import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

thresholds = np.arange(0.1, 0.6, 0.05)

files = os.listdir('thresh/pck')

all_results_pck = pd.DataFrame()
for file in files:
    path = os.path.join('thresh/pck',file)
    results = pd.read_csv(path, header=None)
    all_results_pck = pd.concat([all_results_pck, results], axis = 1)

print(all_results_pck)
pcks = all_results_pck.mean(axis = 1)
print(pcks)


files = os.listdir('thresh/ap')

all_results_ap = pd.DataFrame()
for file in files:
    path = os.path.join('thresh/ap',file)
    results = pd.read_csv(path, header=None)
    all_results_ap = pd.concat([all_results_ap, results], axis = 1)

aps = all_results_ap.mean(axis = 1)
#print(aps)

plt.figure(1)
#plt.title('PCKh')
plt.scatter(thresholds, pcks)
plt.xlabel('Ułamek długości głowy')
plt.ylabel('Średni wynik PCKh')
#plt.plot(thresholds, res[0]*thresholds + res[1],'r--')

#res = linregress(thresholds, ap_list)
plt.figure(2)
#plt.title('mAP')
plt.scatter(thresholds, aps)
plt.xlabel('Ułamek długości głowy')
plt.ylabel('mAP')
#plt.plot(thresholds, res[0]*thresholds + res[1],'r--')

plt.show()