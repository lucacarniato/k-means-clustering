<<<<<<< HEAD
#if defined(_WIN32) 
#define DISTANCE_CALCULATOR_API __declspec(dllexport)
#else  
#define DISTANCE_CALCULATOR_API __attribute__((visibility("default")))
#endif

#include "omp.h"

// contains all mesh instances
namespace DistanceCalculator
{
    extern "C"
    {

        DISTANCE_CALCULATOR_API void distance_calculator(
            int num_clusters,
            int num_points,
            int num_dimensions,
            int num_threads,
            double* centroids_coordinates,
            double* points,
            double* min_distance,
            int* min_distance_index)
        {
            if (num_threads < 1)
            {
                return;
            }

            // set the numebr of threads
            omp_set_num_threads(num_threads);

#pragma omp parallel for
            for (int p = 0; p < num_points; ++p)
            {
                int point_position = p * num_dimensions;

                for (int c = 0; c < num_clusters; ++c)
                {
                    int cluster_position = c * 2;
                    double squared_distance = 0.0;
                    for (int d = 0; d < num_dimensions; ++d)
                    {
                        double delta = centroids_coordinates[cluster_position + d] - points[point_position + d];
                        squared_distance += delta * delta;

                    }

                    if (squared_distance < min_distance[p])
                    {
                        min_distance[p] = squared_distance;
                        min_distance_index[p] = c;
                    }

                }

            }

        }

    }
}
=======
#if defined(_WIN32) 
#define DISTANCE_CALCULATOR_API __declspec(dllexport)
#else  
#define DISTANCE_CALCULATOR_API __attribute__((visibility("default")))
#endif

#include "omp.h"

// contains all mesh instances
namespace DistanceCalculator
{
    extern "C"
    {

        DISTANCE_CALCULATOR_API void distance_calculator(
            int num_clusters,
            int num_points,
            int num_dimensions,
            int num_threads,
            double* centroids_coordinates,
            double* points,
            double* min_distance,
            int* min_distance_index)
        {
            if (num_threads < 1)
            {
                return;
            }

            // set the numebr of threads
            omp_set_num_threads(num_threads);

#pragma omp parallel for
            for (int p = 0; p < num_points; ++p)
            {
                int point_position = p * num_dimensions;

                for (int c = 0; c < num_clusters; ++c)
                {
                    int cluster_position = c * 2;
                    double squared_distance = 0.0;
                    for (int d = 0; d < num_dimensions; ++d)
                    {
                        double delta = centroids_coordinates[cluster_position + d] - points[point_position + d];
                        squared_distance += delta * delta;

                    }

                    if (squared_distance < min_distance[p])
                    {
                        min_distance[p] = squared_distance;
                        min_distance_index[p] = c;
                    }

                }

            }

        }

    }
}
>>>>>>> b60edd4f708b751cf6a25be6160dc4cd7edbcef1
