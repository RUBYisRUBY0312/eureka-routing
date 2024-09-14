from sklearn.cluster import KMeans


def create_big_clusters(num_clusters, customer_euclidean_coords):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit_predict(customer_euclidean_coords)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    return centroids, labels


def create_smaller_clusters(customer_euclidean_coords, cluster_demands, vehicle_capacity, min_customer_per_subcluster):
    num_smaller_clusters = len(customer_euclidean_coords) // min_customer_per_subcluster  # Adjust the number of smaller clusters
    if num_smaller_clusters == 0:
        num_smaller_clusters = 1

    kmeans = KMeans(n_clusters=num_smaller_clusters)
    smaller_clusters = kmeans.fit_predict(customer_euclidean_coords)

    # Initialize a list to store the sum of demands for each smaller cluster
    demand_sum_per_cluster = [0] * num_smaller_clusters

    # Assign customers to smaller clusters ensuring the capacity constraint
    for i, cluster_id in enumerate(smaller_clusters):
        demand_sum_per_cluster[cluster_id] += cluster_demands[i]
        if demand_sum_per_cluster[cluster_id] > vehicle_capacity:
            # Reassign the customer to another cluster if the capacity is exceeded
            for new_cluster_id in range(num_smaller_clusters):
                if demand_sum_per_cluster[new_cluster_id] + cluster_demands[i] <= vehicle_capacity:
                    smaller_clusters[i] = new_cluster_id
                    demand_sum_per_cluster[new_cluster_id] += cluster_demands[i]
                    demand_sum_per_cluster[cluster_id] -= cluster_demands[i]
                    break

    return smaller_clusters, demand_sum_per_cluster