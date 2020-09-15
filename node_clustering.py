import dill
#import plot
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi 
from sklearn.metrics import adjusted_rand_score as ari 
import torch
import pandas as pd 
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
#acm_gtn_result_dict.pk acm_result_dict.pk acm_gcn_result_dict acm_han_result_dict
filename = 'acm_result_dict.pk'
#filename = 'imdb_result_dict.pk'
result = dill.load(open(filename,'rb'))
n_class = result['y_pred'].size()[1]
y_pred = torch.argmax(result['y_pred'],dim=-1)
y_true = result['y_true']
embeddings = result['embedding'].cpu().numpy()
print(y_pred.size())
print(y_true.size())
print(embeddings.shape)
'''
    caculate NMI and ARI
    normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
    adjusted_rand_score(labels_true, labels_pred)[source]
'''
kmeans = KMeans(n_clusters=n_class, random_state=0).fit(embeddings)
y_cluster = kmeans.labels_

nmi_score = nmi(y_true.cpu().numpy(), y_cluster)
ari_score = ari(y_true.cpu().numpy(), y_cluster)
print('nmi_score={}, ari_score={}'.format(nmi_score, ari_score))

'''
    visualize (using t-SNE)
'''
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=500)
tsne_results = tsne.fit_transform(embeddings)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
df_subset = pd.DataFrame(index=list(range(embeddings.shape[0])), columns=['axis_x', 'axis_y'])
df_subset['axis_x'] = tsne_results[:,0]
df_subset['axis_y'] = tsne_results[:,1]
df_subset['y'] = y_true.cpu().numpy()

# plt.figure(figsize=(16,7))
plt.figure()

ax = plt.subplot()
sns.scatterplot(
    x="axis_x", y="axis_y",
    hue="y",
    palette=sns.color_palette("hls", n_class),
    data=df_subset,
    legend="full",
    alpha=0.3,
    ax=ax
)
plt.savefig('{}_visualization.png'.format(filename))
