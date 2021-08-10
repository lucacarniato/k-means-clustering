import math
import os
import random
from ctypes import CDLL, POINTER, c_double, c_int, cast
from multiprocessing import Process
from multiprocessing.sharedctypes import RawArray

import matplotlib.pyplot as plt
import numpy as np


def assign_points_to_cluster(
    num_dimension,
    num_clusters,
    start_index,
    end_index,
    centroids_coordinates,
    points,
    squared_distances,
    min_distance_index,
):
    for i in range(start_index, end_index):
        point_index = i * num_dimension
        min_distance_index[i] = -1
        squared_distances[i] = math.inf
        for j in range(num_clusters):

            cluster_index = j * num_dimension
            dis = 0.0
            for d in range(num_dimension):
                delta = (
                    points[point_index + d] - centroids_coordinates[cluster_index + d]
                )
                dis += delta * delta

            if dis < squared_distances[i]:
                min_distance_index[i] = j
                squared_distances[i] = dis


class KMeansMultiProcess:
    def __init__(
        self,
        num_centroids,
        input_points,
        num_iterations,
        num_processes,
        external_kernel=False,
        random_state=None,
    ):

        if input_points.shape[0] < num_centroids:
            raise ValueError(
                "Number of clusters k={} is smaller than number of data points n={}".format(
                    self.num_clusters, self.num_points
                )
            )
        if input_points.shape[1] < 2:
            raise ValueError("data points must have at least two dimensions")

        self.num_iterations = num_iterations
        self.num_clusters = num_centroids
        self.num_dimension = input_points.shape[1]
        self.num_points = input_points.shape[0]
        self.num_processes = num_processes
        self.random_state = random_state

        self.centroids = np.full(
            self.num_clusters * self.num_dimension, np.inf, dtype=np.double
        )
        self.point_class = np.full(self.num_points, -1, dtype=np.int)
        self.cluster_energy = np.zeros(self.num_clusters)
        self.point_distances = np.zeros(self.num_points)
        self.clusters_size = np.zeros(self.num_clusters, dtype=np.int)
        self.clusters_sum = np.full(
            self.num_clusters * self.num_dimension, 0.0, dtype=np.double
        )

        if external_kernel:
            self.points = np.full(
                self.num_points * self.num_dimension, np.inf, dtype=np.double
            )
            self.squared_distances = np.full(self.num_points, math.inf, dtype=np.double)
            self.min_distance_index = np.full(self.num_points, -1, dtype=np.int)
        else:
            self.points = RawArray(
                "d",
                np.full(self.num_points * self.num_dimension, np.inf, dtype=np.double),
            )
            self.squared_distances = RawArray(
                "d", np.full(self.num_points, math.inf, dtype=np.double)
            )
            self.min_distance_index = RawArray(
                "i", np.full(self.num_points, -1, dtype=np.int)
            )

        for i, point in enumerate(input_points):
            point_index = i * self.num_dimension
            for d in range(self.num_dimension):
                self.points[point_index + d] = point[d]

        self._initialize_centroids_k_means_pp()
        self.external_kernel = external_kernel

        if external_kernel:
            current_working_dir = os.getcwd()
            dll_name = "DistanceCalculator.dll"
            dll_path = os.path.join(current_working_dir, dll_name)
            self.kernelDll = CDLL(dll_path)
            print(self.kernelDll)

    def _initialize_centroids_k_means_pp(self):

        # k-means plus plus
        if self.random_state:
            random.seed(self.random_state)
        first_index = random.randint(0, self.num_points)

        # first centroid
        point_index = first_index * self.num_dimension
        for d in range(self.num_dimension):
            self.centroids[d] = self.points[point_index + d]

        for cluster_index in range(1, self.num_clusters):

            assign_points_to_cluster(
                self.num_dimension,
                cluster_index,
                0,
                self.num_points,
                self.centroids,
                self.points,
                self.squared_distances,
                self.min_distance_index,
            )

            distances = np.sqrt(self.squared_distances)
            sum_distances = np.sum(distances)
            normalized_distances = distances / sum_distances

            r = random.uniform(0, 1)
            acc = 0
            chosen_index = 0
            for n in normalized_distances:
                acc += n
                if acc >= r:
                    break
                chosen_index += 1

            cluster_index = cluster_index * self.num_dimension
            point_index = chosen_index * self.num_dimension
            for d in range(self.num_dimension):
                self.centroids[cluster_index + d] = self.points[point_index + d]

    def _spawn_process(self):

        self.processes = []
        num_points_per_thread = math.ceil(self.num_points / self.num_processes)
        start_index = 0
        for num_process in range(self.num_processes):
            end_index = start_index + num_points_per_thread
            if end_index > self.num_points:
                end_index = self.num_points
            self.processes.append(
                Process(
                    target=assign_points_to_cluster,
                    args=(
                        self.num_dimension,
                        self.num_clusters,
                        start_index,
                        end_index,
                        self.centroids,
                        self.points,
                        self.squared_distances,
                        self.min_distance_index,
                    ),
                )
            )

            start_index = end_index + 1

        assert end_index == self.num_points

    def _start_process(self):
        for p in self.processes:
            p.start()

    def _join_process(self):
        for p in self.processes:
            p.join()

    def _compute_distances(self):

        if self.external_kernel:

            centroids_pointer = cast(self.centroids.ctypes.data, POINTER(c_double))
            points_pointer = cast(self.points.ctypes.data, POINTER(c_double))
            min_distance_pointer = cast(
                self.squared_distances.ctypes.data, POINTER(c_double)
            )
            min_distance_index_pointer = cast(
                self.min_distance_index.ctypes.data, POINTER(c_int)
            )

            # invoke the external distance calculator function
            self.kernelDll.distance_calculator(
                c_int(self.num_clusters),
                c_int(self.num_points),
                c_int(self.num_dimension),
                c_int(self.num_processes),
                centroids_pointer,
                points_pointer,
                min_distance_pointer,
                min_distance_index_pointer,
            )
        else:
            # spawn workers
            self._spawn_process()
            self._start_process()
            self._join_process()

    def fit(self):

        for iteration in range(self.num_iterations):

            self._compute_distances()

            for i in range(self.num_points):
                # subtract the current point from previous cluster
                if self.min_distance_index[i] != self.point_class[i]:
                    if self.point_class[i] > 0:
                        self.clusters_size[self.point_class[i]] -= 1
                        self.cluster_energy[
                            self.point_class[i]
                        ] -= self.point_distances[i]
                        cluster_index = self.point_class[i] * self.num_dimension
                        point_index = i * self.num_dimension
                        for d in range(self.num_dimension):
                            self.clusters_sum[cluster_index + d] -= self.points[
                                point_index + d
                            ]

                    # assign new cluster and distances
                    self.point_class[i] = self.min_distance_index[i]
                    self.clusters_size[self.min_distance_index[i]] += 1

                    cluster_index = self.point_class[i] * self.num_dimension
                    point_index = i * self.num_dimension
                    for d in range(self.num_dimension):
                        self.clusters_sum[cluster_index + d] += self.points[
                            point_index + d
                        ]

                    self.point_distances[i] = self.squared_distances[i]
                    self.cluster_energy[
                        self.min_distance_index[i]
                    ] += self.point_distances[i]

            # update centroids
            diff = 0.0
            for i in range(self.num_clusters):
                cluster_index = i * self.num_dimension
                for d in range(self.num_dimension):
                    new_coordinate = (
                        self.clusters_sum[cluster_index + d] / self.clusters_size[i]
                    )
                    diff = max(
                        diff, abs(self.centroids[cluster_index + d] - new_coordinate)
                    )
                    self.centroids[cluster_index + d] = new_coordinate

            # break iteration
            if diff < 1e-12:
                print("Centroids unchanged at iteration ", iteration, " terminating...")
                break


def plot_strong_scaling(n_samples, num_clusters, max_num_processes):
    global kMeansMultiProcess
    from sklearn.datasets.samples_generator import make_blobs

    input_points, y_values = make_blobs(
        n_samples=n_samples, centers=num_clusters, cluster_std=0.4, random_state=42
    )

    times_multi_processes_python = np.empty((max_num_processes,))
    for num_processes in range(1, max_num_processes + 1):
        print("num_processes ", num_processes)
        kMeansMultiProcess = KMeansMultiProcess(
            num_clusters,
            input_points,
            num_iterations=100,
            num_processes=num_processes,
            random_state=42,
            external_kernel=False,
        )
        from timeit import timeit

        times_multi_processes_python[num_processes - 1] = timeit(
            "kMeansMultiProcess.fit()", number=1, globals=globals()
        )
    print(times_multi_processes_python)

    times_multi_threaded_external = np.empty((max_num_processes,))
    for num_processes in range(1, max_num_processes + 1):
        print(num_processes)
        kMeansMultiProcess = KMeansMultiProcess(
            num_clusters,
            input_points,
            num_iterations=100,
            num_processes=num_processes,
            random_state=42,
            external_kernel=True,
        )
        from timeit import timeit

        times_multi_threaded_external[num_processes - 1] = timeit(
            "kMeansMultiProcess.fit()", number=1, globals=globals()
        )
    # print(times_multi_threaded_external)

    plt.figure(figsize=(10, 4))
    ax = plt.axes()
    threads = range(1, max_num_processes + 1)
    ax.plot(threads, times_multi_processes_python, "r--", label="Python multiprocess")
    ax.plot(
        threads,
        times_multi_threaded_external,
        "b--",
        label="C++ external library with OpenMP",
    )
    plt.title(
        "Strong scaling with "
        + str(n_samples)
        + " samples, "
        + str(num_clusters)
        + " clusters",
        fontsize=14,
    )
    plt.xlabel("Number of processes", fontsize=16)
    plt.ylabel("Wall clock time (s)", fontsize=16)
    plt.xlim(1, max_num_processes)
    plt.ylim(0, 20)
    import matplotlib.ticker as mticker

    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.legend(loc=2)


def plot_results(n_samples, num_clusters, num_processes, external_kernel):
    from sklearn.datasets.samples_generator import make_blobs

    input_points, labels, true_centroids = make_blobs(
        n_samples=n_samples,
        centers=num_clusters,
        cluster_std=0.4,
        random_state=0,
        return_centers=True,
    )
    kMeansMultiProcess = KMeansMultiProcess(
        num_clusters,
        input_points,
        num_iterations=100,
        num_processes=num_processes,
        external_kernel=external_kernel,
    )
    kMeansMultiProcess.fit()

    print("Plotting...")
    cross_color = "k"
    calculated_centroids = np.reshape(
        kMeansMultiProcess.centroids,
        (kMeansMultiProcess.num_clusters, kMeansMultiProcess.num_dimension),
    )

    plt.scatter(
        calculated_centroids[:, 0],
        calculated_centroids[:, 1],
        marker="x",
        s=50,
        linewidths=50,
        color=cross_color,
        zorder=11,
        alpha=1,
    )

    plt.scatter(
        true_centroids[:, 0],
        true_centroids[:, 1],
        marker="+",
        s=50,
        linewidths=50,
        color="r",
        zorder=11,
        alpha=1,
    )

    plt.plot(input_points[:, 0], input_points[:, 1], "k.", markersize=2)
