
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 导入模块
plt.rcParams['font.sans-serif']=['Times New Roma'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
sns.set() # 设置美化参数，一般默认就好
df=pd.read_csv('smooth.csv')

sns.lineplot(x="BC_frame",y="BC_reward",data=df,color='#3CB44B')
sns.lineplot(x="AWAC_frame",y="AWAC_reward",data=df,color='#E6194B')
sns.lineplot(x="BCQ_frame",y="BCQ_reward",data=df,color='#FFE110')
sns.lineplot(x="CQL_frame",y="CQL_reward",data=df,color='#4363D8')
sns.lineplot(x="CRR_frame",y="CRR_reward",data=df,color='#F58231')
sns.lineplot(x="IQL_frame",y="IQL_reward",data=df,color='#911EB4')
sns.lineplot(x="PLAS_frame",y="PLAS_reward",data=df,color='#42D4F4')
sns.lineplot(x="TD3+BC_frame",y="TD3+BC_reward",data=df,color='#BFEF45')
sns.lineplot(x="DOGE_frame",y="DOGE_reward",data=df,color='#469990')
sns.lineplot(x="VEM_frame",y="VEM_reward",data=df,color='#9A6324')
plt.xlabel("Training Epoch 100")
plt.ylabel("Epoch Reward")
plt.legend(labels=["BC","AWAC","BCQ","CQL","CRR","IQL","PLAS","TD3+BC","DOGE","VEM"])
# plt.title("Expert dataste")

plt.show()




