import json, os
import matplotlib.pyplot as plt
import shutil

with open('./MAE_result.json','r') as f:
    data = json.load(f)
select_pat = []
for pat in data.keys():
    select_pat.append(pat)

pat_list = os.listdir('./Dataset_rep/')
if not os.path.exists('./Dataset_select'):
    os.mkdir('./Dataset_select')

# for pat in pat_list:
#     if pat.split('.')[0] in select_pat:
#         src_file = './Dataset_rep/' + pat
#         shutil.copy(src_file,'./Dataset_select')

print(len(os.listdir('./Dataset_select')))
# print(len(pat_list))
# print(len(data))
# dbp_err =[]
# sbp_err =[]
# pat = []
# for key,val in data.items():
#     pat.append(key)
#     dbp_err.append(val['DBP']['Test'])
#     sbp_err.append(val['SBP']['Test'])

# fig,axs = plt.subplots(1,2)
# axs[0].hist(dbp_err, bins=20, rwidth=0.8)
# axs[0].set_xlabel('Error(mmHg)')
# axs[0].set_ylabel('Frequencies')
# axs[0].set_title('Histogram of error DBP')
# axs[0].grid()

# axs[1].hist(sbp_err, bins=20, rwidth=0.8)
# axs[1].set_xlabel('Error(mmHg)')
# axs[1].set_ylabel('Frequencies')
# axs[1].set_title('Histogram of error SBP')
# axs[1].grid()
# plt.show()