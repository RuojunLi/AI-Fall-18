import numpy as np
from random import random
from scipy.spatial import distance
import  matplotlib.pyplot as plt

################################################
# Assign each point to nearest cluster by calculating
# its distance to each centroid.
# Find new cluster center by taking the average of
# the assigned points.
def cluster_Kmeans(data,cluster_num):
    cluster_centroids = []
    cluster_centroids_new = [[] for x in range(cluster_num)]
    for i in range(cluster_num):
        centroid_temp = np.array([random(), random()]).reshape(1, 2)
        cluster_centroids.append(centroid_temp)
    for i in range(cluster_num):
        centroid_temp = np.array([random(), random()]).reshape(1, 2)
        cluster_centroids.append(centroid_temp)

    while True:
        cluster_distance = []
        cluster_dict = [[] for x in range(cluster_num)]
        for i in range(cluster_num):
            centroid_temp = cluster_centroids[i]
            print("Cluster[",i,"]: ",centroid_temp)
            cluster_distance.append(distance.cdist(centroid_temp, data, 'euclidean').reshape(1, len(data)))

        for i in range(len(data)):
            distance_i = [cluster_distance[cluster_index][0,i] for cluster_index in range(cluster_num)]
            (m, index) = min((v, index) for index, v in enumerate(distance_i))
            cluster_dict[index].append(i)

        for i in range(cluster_num):
            cluster_temp = cluster_dict[i]
            data_temp = data[cluster_temp,0:2]
            cluster_centroids_new[i] = np.asarray([np.mean(data_temp[:,0]),np.mean(data_temp[:,1])]).reshape(1,2)
        if cluster_centroids_new == cluster_centroids:
            print(cluster_centroids)
            break
        else:
            cluster_centroids = cluster_centroids_new
    return cluster_dict, cluster_centroids

###########################
#  load file
file = open("cluster_data.txt", "r")
data = np.array([np.array([float(data_line.split()[1]),
                           float(data_line.split()[2])])
                 for data_line in file.readlines()])

###############################################
#Intialize the parameters and cluster centroids
cluster_num = 3
cluster_color = ['ro','g^','bs']
cluster_dict,cluster_centroids = cluster_Kmeans(data,cluster_num)



#################################################
#Plot: shows the cluplt.show()ster with different colors:
for i in range(cluster_num):
    cluster_temp = cluster_dict[i]
    data_temp = data[cluster_temp, 0:2]
    plt.plot(data_temp[:,0],data_temp[:,1],cluster_color[i])
    plt.plot(cluster_centroids[i][0,0],cluster_centroids[i][0,1],cluster_color[i-1],markersize=20)


