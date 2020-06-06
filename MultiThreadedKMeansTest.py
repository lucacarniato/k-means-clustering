from MultiThreadedKMeans import *
import unittest

class MultiThreadedKMeansUnitTest(unittest.TestCase):

    def test_assign_points_to_cluster(self):
        #this is testing the stateless function test_assign_points_to_cluster
        num_clusters = 2
        num_points = 3
        start_index = 0
        end_index =3
        centroids = np.zeros((num_clusters, 2))
        centroids[0, :] = [3.0, 3.0]
        centroids[1, :] = [7.0, 1.0]
        points = np.zeros((num_points, 2))
        points[0, :] = [3.0, 4.0]
        points[1, :] = [5.0, 3.0]
        points[2, :] = [8.0, 1.0]
        min_distance =  np.full(num_points, math.inf, dtype= np.float64)
        min_distance_index = np.full(num_points, float('nan'), dtype=np.int16)
        kMeansParallelAlgorithm = KMeansMultiThreaded()
        kMeansParallelAlgorithm.assign_points_to_cluster(num_clusters,
                                                        start_index,
                                                        end_index,
                                                        centroids,
                                                        points,
                                                        min_distance,
                                                        min_distance_index)

        self.assertEqual(min_distance[0], 1.0,'incorrect distance points[0, :]')
        self.assertEqual(min_distance[1], 2.0, 'incorrect distance points[1, :]')
        self.assertEqual(min_distance[2], 1.0, 'incorrect distance points[2, :]')


    def test_centroid_positions(self):
        input_points = pd.read_csv('pts.csv', names=['X', 'Y'])
        kMeansParallelAlgorithm = KMeansMultiThreaded()
        kMeansParallelAlgorithm.Set(num_centroids = 10, input_points = input_points.values, num_iterations=100, num_processes=4)
        kMeansParallelAlgorithm.fit()

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[0,0], 0.04521788, 7, 'incorrect centroids[0 ,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[0,1], -0.06817853, 7, 'incorrect centroids[0, 1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[1, 0], -0.12789202, 7, 'incorrect centroids[1,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[1, 1], 1.47766361, 7, 'incorrect centroids[1,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[2, 0], 0.06761798, 7, 'incorrect centroids[2,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[2, 1], -0.93269577, 7, 'incorrect centroids[2,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[3, 0], -0.14027231, 7, 'incorrect centroids[3,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[3, 1], 0.5203904, 7, 'incorrect centroids[3,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[4, 0], -0.65129033, 7, 'incorrect centroids[4,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[4, 1], -0.51177182, 7, 'incorrect centroids[4,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[5, 0], 0.67148286, 7, 'incorrect centroids[5,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[5, 1], 0.75762594, 7, 'incorrect centroids[5,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[6, 0], 0.82854806, 7, 'incorrect centroids[6,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[6, 1], -0.23387842, 7, 'incorrect centroids[6,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[7, 0], 0.62301066, 7, 'incorrect centroids[7,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[7, 1],  -1.38096835, 7, 'incorrect centroids[7,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[8, 0], -0.78295523, 7, 'incorrect centroids[8,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[8, 1], -1.18216902, 7, 'incorrect centroids[8,1]')

        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[9, 0], -0.85165581, 7, 'incorrect centroids[0,0]')
        self.assertAlmostEqual(kMeansParallelAlgorithm.centroids[9, 1], 0.37766087, 7, 'incorrect centroids[0,1]')

if __name__ == '__main__':
    unittest.main()