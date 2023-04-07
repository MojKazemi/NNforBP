import json
import matplotlib.pyplot as plt

with open('./my_dict.json','r') as f:
    data = json.load(f)
dbp_err =[]
sbp_err =[]
pat = []
for key,val in data.items():
    pat.append(key)
    dbp_err.append(val['DBP']['Test'])
    sbp_err.append(val['SBP']['Test'])

fig,axs = plt.subplots(1,2)
axs[0].hist(dbp_err, bins=20, rwidth=0.8)
axs[0].set_xlabel('Error(mmHg)')
axs[0].set_ylabel('Frequencies')
axs[0].set_title('Histogram of error DBP')
axs[0].grid()

axs[1].hist(sbp_err, bins=20, rwidth=0.8)
axs[1].set_xlabel('Error(mmHg)')
axs[1].set_ylabel('Frequencies')
axs[1].set_title('Histogram of error SBP')
axs[1].grid()
plt.show()