import random
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from threading import Thread

class KMeansMultiThreaded:

    def Set(self, num_centroids, input_points, num_iterations, num_processes):

        self.num_clusters = num_centroids
        self.points = input_points

        self.num_dimension = input_points.shape[1]
        self.num_points = input_points.shape[0]
        self.num_iterations = num_iterations
        self.num_processes = num_processes
        self.centroids = np.zeros((num_centroids, self.num_dimension))
        self.point_class = np.full(self.num_points, -1, dtype=np.int16)
        self.cluster_energy = np.zeros(self.num_clusters)
        self.point_distances = np.zeros(self.num_points )
        self.clusters_size = np.zeros(self.num_clusters, dtype=np.int32)
        self.clusters_sum = np.zeros((self.num_clusters, self.num_dimension))

        if self.num_points < self.num_clusters:
            raise ValueError("Number of clusters k={} is smaller than number of data points n={}".format(self.num_clusters, self.num_points))
        if self.num_dimension < 2:
            raise ValueError("data points must have at least two dimensions")

        self.initialize_centroids_k_means_pp()

    def assign_points_to_cluster(self, num_clusters, start_index, end_index, centroids, points, min_distance, min_distance_index):

        if num_clusters > len(centroids):
            return

        for i in range(start_index, end_index):
            for j in range(num_clusters):
                d = np.linalg.norm(points[i] - centroids[j])
                if d < min_distance[i]:
                    min_distance_index[i] = j
                    min_distance[i] = d

    def initialize_centroids(self):
        #pick up points at random to initialize the centroids
        random_indexes = list(range(self.num_points))
        random.seed(42)
        random.shuffle(random_indexes)
        for i in range(self.num_clusters):
            self.centroids[i] = self.points[random_indexes[i]]

    def initialize_centroids_k_means_pp(self):
        # k-means plus plus
        random.seed(42)
        self.centroids[0] = random.choice(self.points)
        for cluster_index in range(1, self.num_clusters):

            self.min_distance = np.full(self.num_points, math.inf, dtype=np.float64)
            self.min_distance_index = np.full(self.num_points, float('nan'), dtype=np.int16)
            self.assign_points_to_cluster(cluster_index + 1, 0, len(self.points), self.centroids, self.points, self.min_distance, self.min_distance_index)
            sum_distances = np.sum(self.min_distance)
            normalized_distances = self.min_distance / sum_distances
            np.random.seed(42)
            r = np.random.uniform()
            acc = 0
            chosen_index = 0
            for n in normalized_distances:
                acc += n
                if acc >= r:
                    break
                chosen_index += 1

            self.centroids[cluster_index] = self.points[chosen_index]


    def spawn_threads(self):

        self.min_distance =  np.full(self.num_points, math.inf, dtype= np.float64)
        self.min_distance_index = np.full(self.num_points, float('nan'), dtype=np.int16)
        self.threads = []
        num_points_per_thread = math.ceil(len(self.points)/self.num_processes)
        start_index = 0
        for num_process in range(self.num_processes):
            end_index = start_index + num_points_per_thread
            if end_index > len(self.points):
                end_index = len(self.points)
            self.threads.append(Thread(target=self.assign_points_to_cluster, args=(self.num_clusters, start_index, end_index,
                                                                                   self.centroids, self.points, self.min_distance, self.min_distance_index)))
            start_index = end_index + 1

        assert(end_index==len(self.points))


    def start_threds(self):
        for t in self.threads:
            t.start()

    def join_threds(self):
        for t in self.threads:
            t.join()

    def fit(self):

        for iteration in range(self.num_iterations):

            #spawn workers
            self.spawn_threads()
            self.start_threds()
            self.join_threds()

            for i in range(len(self.points)):
                # subtract the current point from previous cluster
                if self.min_distance_index[i] != self.point_class[i]:
                    if self.point_class[i] != -1:
                        self.clusters_size[self.point_class[i]] -= 1
                        self.clusters_sum[self.point_class[i]] -= self.points[i]
                        self.cluster_energy[self.point_class[i]] -= self.point_distances[i]

                    #assign new cluster and distances
                    self.point_class[i] = self.min_distance_index[i]
                    self.clusters_size[self.min_distance_index[i]] += 1
                    self.clusters_sum[self.min_distance_index[i]] += self.points[i]
                    self.point_distances[i] =self.min_distance[i] ** 2
                    self.cluster_energy[self.min_distance_index[i]] += self.point_distances[i]

            # update centroids if needed
            centroids_position_changed = True
            for i in range(self.num_clusters):
                new_centroid = self.clusters_sum[i] / self.clusters_size[i]
                centroids_position_changed = centroids_position_changed and np.array_equal(new_centroid, self.centroids[i])
                self.centroids[i] = new_centroid

            # print("iteration ",iteration, " completed ", self.centroids[0])
            if centroids_position_changed:
                print("Centroids unchanged at iteration ", iteration ," terminating...")
                break

def plot_strong_scaling(num_centroids, max_num_threads):
    global kMeansParallelAlgorithm
    input_points = pd.read_csv('pts.csv', names=['X', 'Y'])
    times = np.empty((max_num_threads,))
    for num_processes in range(1, max_num_threads + 1):
        print(num_processes)
        kMeansParallelAlgorithm = KMeansMultiThreaded()
        kMeansParallelAlgorithm.Set(num_centroids, input_points.values, num_iterations = 100, num_processes = num_processes)
        from timeit import timeit
        times[num_processes-1] = timeit("kMeansParallelAlgorithm.fit()", number=3,globals=globals())

    print(times)

    plt.figure(figsize=(10, 4))
    ax = plt.axes()
    threads = range(1, max_num_threads + 1)
    ax.plot(threads, times, "r--", label="Wall clock time (s)")
    plt.title("Scaling", fontsize=14)
    plt.xlabel("Number of threads", fontsize=16)
    plt.ylabel("Wall clock time (s)", fontsize=16)
    plt.xlim(1, max_num_threads)
    plt.ylim(0, 1)

def plot_results(num_clusters, num_processes):

    input_points = pd.read_csv('pts.csv', names=['X', 'Y'])
    kMeansParallelAlgorithm = KMeansMultiThreaded()
    kMeansParallelAlgorithm.Set(num_clusters, input_points.values, num_iterations=100, num_processes=num_processes)
    kMeansParallelAlgorithm.fit()

    print("Plotting...")
    circle_color ='w'
    cross_color ='k'
    plt.scatter(kMeansParallelAlgorithm.centroids[:, 0], kMeansParallelAlgorithm.centroids[:, 1],
                marker='o', s=30, linewidths=8,color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(kMeansParallelAlgorithm.centroids[:, 0], kMeansParallelAlgorithm.centroids[:, 1],
                marker='x', s=50, linewidths=50,color=cross_color, zorder=11, alpha=1)

    plt.plot(kMeansParallelAlgorithm.points[:, 0], kMeansParallelAlgorithm.points[:, 1], 'k.', markersize=2)
