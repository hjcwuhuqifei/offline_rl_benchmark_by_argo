import matplotlib.pyplot as plt
from sklearn import manifold, datasets
import pickle
import numpy as np
 
"""对S型曲线数据的降维和可视化"""
x, color = datasets._samples_generator.make_s_curve(n_samples=1000, random_state=0)

dataset  = pickle.load(open('dataset_expert', 'rb'))
dataset_from_td3 = pickle.load(open( 'dataset_from_td3_10', 'rb'))
dataset_from_td3_plus_expert = pickle.load(open( 'dataset_expert_plus_td3', 'rb'))
dataset_expert_plus_random = pickle.load(open( 'dataset_expert_plus_random', 'rb'))

x_expert = dataset['observations']
x_from_td3 = np.array(np.squeeze(dataset_from_td3['state']))
x_td3_plus_expert = dataset_from_td3_plus_expert['observations']
x_expert_plus_random = dataset_expert_plus_random['observations']

n_neighbors = 10
n_components = 2
 
# 创建自定义图像
fig = plt.figure(figsize=(8, 8))		# 指定图像的宽和高
 
# # 绘制S型曲线的3D图像
# ax = fig.add_subplot(211, projection='3d')		# 创建子图
# ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)		# 绘制散点图，为不同标签的点赋予不同的颜色
# ax.set_title('Original S-Curve', fontsize=14)
# ax.view_init(4, -72)		# 初始化视角
 
# t-SNE的降维与可视化
ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# 训练模型
y_expert = ts.fit_transform(x_expert)
y_from_td3 = ts.fit_transform(x_from_td3)
y_td3_plus_expert = ts.fit_transform(x_td3_plus_expert)
y_expert_plus_expert = ts.fit_transform(x_expert_plus_random)

plt.scatter(y_expert[:, 0], y_expert[:, 1], c='red', cmap=plt.cm.Spectral)
plt.scatter(y_from_td3[:, 0], y_from_td3[:, 1], c='blue', cmap=plt.cm.Spectral)
plt.scatter(y_td3_plus_expert[:, 0], y_td3_plus_expert[:, 1], c='yellow', cmap=plt.cm.Spectral)
plt.scatter(y_expert_plus_expert[:, 0], y_expert_plus_expert[:, 1], c='green', cmap=plt.cm.Spectral)
# ax1.set_title('t-SNE Curve', fontsize=14)
# 显示图像
plt.savefig('save_figure.png')
