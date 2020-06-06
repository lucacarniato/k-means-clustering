# Multi threaded K-Means clustering

This project implements the KMeans clustering algorithm using multiple threads. The following steps are followed:

* Initialize the centroids for the predefined number of clusters.
* Calculate the distances of each point from the centroids and assign each point to the closest centroid. All points assigned to the same centroid represent a cluster. This part of the algorithm is the most computing intensive and was parallelized.
* Updated the coordinates of the centroids with the mean x and y values of all points in the cluster. 
* Iterate from step 2 to 4 until the position of the centroids does not change or the maximum number of iterations is reached.

Centroid initialization follows the k-means++ algorithm (http://ilpubs.stanford.edu:8090/778/), where the centroids positions are initialized far from each other. 

As can be seen below, the multithreaded python implantation does not scale well, due to the GIL, which assigns the python interpreter to a single thread. 
