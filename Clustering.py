import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from sklearn.metrics import silhouette_score

# function to compute hopkins's statistic for the dataframe X
def hopkins_statistic(X):
    X = X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures
    
    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    
    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
    
    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    
    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    
    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]
    
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    
    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H

#making the clustering data
data = pd.read_csv("Credit_Card_Dataset.csv")
cluster_data = data.filter(items= ["Credit_Limit", "Avg_Utilization_Ratio", "Pay_on_time", "Months_on_book", "Income_Category", "Dependent_count"])

#decomposing the cluster data into 3 dimensional features 
analyser = PCA(n_components= 3, copy= True)
decomposed_c_data = pd.DataFrame(analyser.fit_transform(cluster_data))

#scaling the data for uniformity
scaler = StandardScaler()
scaled_data = scaler.fit_transform(decomposed_c_data)

h_s_score = hopkins_statistic(X= cluster_data) #0.847667633803714 data can be clustered sufficiently well 
#print(h_s_score)

#using k-means to cluster the cluster_data 
cluster = KMeans(n_clusters= 2)
print(cluster_data)
cluster_data["Cluster"] = cluster.fit_predict(cluster_data)

#labelling the clusters
cluster_summary = cluster_data.groupby("Cluster").agg({"Credit_Limit" : "mean", "Pay_on_time" : "mean", "Avg_Utilization_Ratio" : "mean"}).reset_index()
print(cluster_summary) 

#calculate the cluster score
score = silhouette_score(X= cluster_data, labels= cluster_data["Cluster"]) #0.6015011224485932 clusters are well defined 
print(score)

#label the data accordingly now 
for row in range(cluster_data.shape[0]):
    if cluster_data["Cluster"][row] == 0:
        cluster_data["Cluster"][row] = "High Risk"
    else:
        cluster_data["Cluster"][row] = "Low Risk"

file = pickle.dump(cluster, open("cluster_model", 'wb'))
#plotting the decomposed cluster data
# fig = plt.figure()
# ax = plt.axes(projection= "3d")

# ax.scatter(decomposed_c_data[0], decomposed_c_data[1], decomposed_c_data[2], c= "green", cmap= labels)
# plt.show()

#plotting the clusters 
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(decomposed_c_data[0], decomposed_c_data[1], decomposed_c_data[2], c=labels, cmap='viridis', s=50)
# plt.title("3D Clusters")
# plt.show()