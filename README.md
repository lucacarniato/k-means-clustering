# Multi process K-Means clustering

This project implements the KMeans clustering algorithm using multiple processes. The following steps are the following:

* Initialize the centroids for the predefined number of clusters.
* Calculate the distances of each point from the centroids and assign each point to the closest centroid. 
All points assigned to the same centroid represent a cluster. This part of the algorithm is the most computing intensive and was parallelized.
* Updated the coordinates of the centroids with the mean x and y values of all points in the cluster. 
* Iterate from step 2 to 4 until the position of the centroids does not change or the maximum number of iterations is reached.

Centroid initialization follows the k-means++ algorithm (http://ilpubs.stanford.edu:8090/778/), 
where the centroids positions are initialized as far as possible. 

The multiprocess implementation does not scale well, because large arrays between Python processes needs to be serialized into strings of bytes using pickle. 

![alt text](https://github.com/lucacarniato/Multithreaded_K-Means_clustering/blob/master/WallClockTime.png)

# Acceleration using an external C++ library for distance calculation 

To accelerate the algorithm, the distances from the centroids can be calculated in the C++ function "distance_calculator". This function takes the addresses of the numpy arrays and performs the calculations of the distances. 
The use of a C++ method to calculate the distance gives a considerable acceleration (10 times faster). 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details