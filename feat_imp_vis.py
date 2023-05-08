import pandas as pd
import random
import seaborn as sns
import matplotlib.pylab as plt

df = pd.read_csv('./result/feat_imp.csv',index_col=0)
# sns.violinplot(data=df, orient='h')
sns.boxplot(data=df, orient='h')
# sns.kdeplot(data=df) # use for MAE
plt.show()
