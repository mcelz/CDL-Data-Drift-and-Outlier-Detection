from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import kneighbors_graph
from umap import UMAP
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import entropy
from scipy.special import rel_entr
from numpy.random import seed
from numpy.random import normal
from sklearn.cluster import AgglomerativeClustering

datao = pd.read_csv('../datasets/revised_sub20_data_Ishu_modification.csv')

datao = datao.iloc[:, 4:]



def normalize_between_a_b(col, a = -1, b = 1):
    min_val = np.min(col)
    max_val = np.max(col)
    new_col = col.map(lambda x: ((b-a) * (x - min_val)/(max_val - min_val)) + a)
    return new_col

cols_to_update = list(datao.columns[:12])


for col in cols_to_update:
    datao[col] = normalize_between_a_b(datao[col])


columns_to_categorize = ['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x',
       'gravity.y', 'gravity.z', 'rotationRate.x', 'rotationRate.y',
       'rotationRate.z', 'userAcceleration.x', 'userAcceleration.y',
       'userAcceleration.z']
colums_to_not_categorize = ['type', 'row_num', 'outlier']


space_cols = 25
for col in columns_to_categorize:
    print(f"col = {col},{' ' * (space_cols - len(col))}min = {datao[col].min()}, max = {datao[col].max()}")

df_4_cats = pd.DataFrame()

for col in columns_to_categorize:
    df_4_cats[col] = pd.cut(datao[col], bins=[-1, -0.5, 0, 0.5, 1.0], labels = ['lowest', 'low', 'mid', 'high'], include_lowest=True)

for col in colums_to_not_categorize:
    df_4_cats[col] = datao[col]


from scipy.cluster.hierarchy import dendrogram, linkage


from scipy.cluster.hierarchy import dendrogram, linkage

Z1 = linkage(datao[columns_to_categorize].to_numpy(), method='single', metric='euclidean')
Z2 = linkage(datao[columns_to_categorize].to_numpy(), method='complete', metric='euclidean')
Z3 = linkage(datao[columns_to_categorize].to_numpy(), method='average', metric='euclidean')
Z4 = linkage(datao[columns_to_categorize].to_numpy(), method='ward', metric='euclidean')


labels = list(datao['type'].values)

max_d = len(datao['type'].unique())

df = df_4_cats

from scipy.cluster.hierarchy import fcluster

f1 = fcluster(Z4, max_d, criterion='maxclust')

print(f"Clusters: {f1}")

# ### Sklearn Hierarchical Clustering

# ## Ward with Euclidean (Z4)



Z5 = AgglomerativeClustering(n_clusters=len(datao['type'].unique()), linkage='ward')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)


df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


Z5 = AgglomerativeClustering(n_clusters=5, linkage='ward')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)



df['labels'] = Z5.labels_



df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)


df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ### Taking motion (downstairs, upstairs, run, walk) or no motion (sit and stand) as the only categories so 2 clusters



Z5 = AgglomerativeClustering(n_clusters=2, linkage='ward')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)


df['labels'] = Z5.labels_


df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)


df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ## Average with Euclidean (Z3)

Z5 = AgglomerativeClustering(n_clusters=len(datao['type'].unique()), linkage='average')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# ### Taking sit and stand as one category so 5 clusters

Z5 = AgglomerativeClustering(n_clusters=5, linkage='average')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ### Taking motion (downstairs, upstairs, run, walk) or no motion (sit and stand) as the only categories so 2 clusters

Z5 = AgglomerativeClustering(n_clusters=2, linkage='average')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_


df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ## Complete with Euclidean (Z2)


Z5 = AgglomerativeClustering(n_clusters=len(datao['type'].unique()), linkage='complete')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ### Taking sit and stand as one category so 5 clusters

Z5 = AgglomerativeClustering(n_clusters=5, linkage='complete')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ### Taking motion (downstairs, upstairs, run, walk) or no motion (sit and stand) as the only categories so 2 clusters

Z5 = AgglomerativeClustering(n_clusters=2, linkage='complete')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_


df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ## Digging deep in average linkage with other distance metrics and hyperparameters

# #### Mahattan distance (L1)
Z5 = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='l1')
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# #### Mahattan distance (L1) and connectivity of 2 neighbors

connectivity_2_clusters = kneighbors_graph(datao[columns_to_categorize], n_neighbors=2, include_self=False)

Z5 = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='l1', connectivity=connectivity_2_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df.head()

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# #### Mahattan distance (L1) and connectivity of 5 neighbors
connectivity_5_clusters = kneighbors_graph(datao[columns_to_categorize], n_neighbors=5, include_self=False)


Z5 = AgglomerativeClustering(n_clusters=2, linkage='average', affinity='l1', connectivity=connectivity_5_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_


df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# ### 5 neighbors in aggclustering with 5 connectivity


Z5 = AgglomerativeClustering(n_clusters=5, linkage='average', affinity='l1', connectivity=connectivity_5_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_


df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ### 5 neighbors in aggclustering with 2 connectivity

Z5 = AgglomerativeClustering(n_clusters=5, linkage='average', affinity='l1', connectivity=connectivity_2_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# It is getting there and standing is getting filtered out a bit

# ## Further refinement

# #### connectivity = 2, 5 clusters with L2

Z5 = AgglomerativeClustering(n_clusters=5, linkage='average', affinity='l2', connectivity=connectivity_2_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# #### connectivity = 2, 5 clusters with cosine


Z5 = AgglomerativeClustering(n_clusters=5, linkage='average', affinity='cosine', connectivity=connectivity_2_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)


df['labels'] = Z5.labels_

df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)


# #### connectivity = 2, 5 clusters with manhattan

Z5 = AgglomerativeClustering(n_clusters=5, linkage='average', affinity='l1', connectivity=connectivity_2_clusters)
Z5.fit_predict(datao[columns_to_categorize])
print(Z5.labels_)

df['labels'] = Z5.labels_


df_temp = df[['outlier', 'labels']]
counts = df_temp.groupby('outlier')['labels'].value_counts().unstack().fillna(0)
print(counts)

df_temp = df[['type', 'labels']]
counts = df_temp.groupby('type')['labels'].value_counts().unstack().fillna(0)
print(counts)

# ### Winner: connectivity = 2, 5 clusters with L2