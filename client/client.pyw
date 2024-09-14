from itertools import chain

import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt

from popups.customer_input_popup import customer_input_popup
from services.clustering_service import create_big_clusters, create_smaller_clusters
from services.routing_service import RoutingService
from utils import lat_long_to_euclidean, populate_location_list, euclidean_to_lat_long, haversine_distance

sg.theme('Gray Gray Gray')  # Let's set our own color theme

KEY_DEPOT_LONG = 'K_DEPOT_LONG'
KEY_DEPOT_LAT = 'K_DEPOT_LAT'
KEY_DEPOT_ERROR = 'K_DEPOT_ERROR'
KEY_FILE_IMPORT = 'K_FILE_IMPORT'
KEY_CUSTOMER_EXPORT = 'K_CUSTOMER_EXPORT'
KEY_CUSTOMER_EXPORT_BUTTON = 'K_CUSTOMER_EXPORT_BUTTON'
KEY_CUSTOMERS_ERROR = 'K_CUSTOMERS_ERROR'
KEY_CUSTOMERS_INPUT = 'K_CUSTOMERS_INPUT'
KEY_CUSTOMER_ADD = 'K_CUSTOMER_ADD'
KEY_CUSTOMER_EDIT = 'K_CUSTOMER_EDIT'
KEY_CUSTOMER_DELETE = 'K_CUSTOMER_DELETE'
KEY_EXECUTE = 'K_EXECUTE'
KEY_NUM_BIG_CLUSTERS = 'K_NUM_BIG_CLUSTERS'
KEY_NUM_BIG_CLUSTERS_ERROR = 'K_NUM_BIG_CLUSTERS_ERROR'
KEY_MIN_CUSTOMER_PER_SUBCLUSTER = 'K_MIN_CUSTOMER_PER_SUBCLUSTER'
KEY_MIN_CUSTOMER_PER_SUBCLUSTER_ERROR = 'K_MIN_CUSTOMER_PER_SUBCLUSTER_ERROR'
KEY_VEHICLE_CAPACITY = 'K_VEHICLE_CAPACITY'
KEY_VEHICLE_CAPACITY_ERROR = 'K_VEHICLE_CAPACITY_ERROR'
KEY_BATTERY_CAPACITY = 'K_BATTERY_CAPACITY'
KEY_BATTERY_CAPACITY_ERROR = 'K_BATTERY_CAPACITY_ERROR'
KEY_BATTERY_CONSUMPTION = 'K_BATTERY_CONSUMPTION'
KEY_BATTERY_CONSUMPTION_ERROR = 'K_BATTERY_CONSUMPTION_ERROR'
KEY_TRAVEL_COST = 'K_TRAVEL_COST'
KEY_TRAVEL_COST_ERROR = 'K_TRAVEL_COST_ERROR'
KEY_OPERATION_COST = 'K_OPERATION_COST'
KEY_OPERATION_COST_ERROR = 'K_OPERATION_COST_ERROR'
KEY_RECHARGING_COST = 'K_RECHARGING_COST'
KEY_RECHARGING_COST_ERROR = 'K_RECHARGING_COST_ERROR'
KEY_RESULT = 'K_RESULT'
KEY_PLOT = 'K_PLOT'
KEY_EXPORT_FILE = 'K_EXPORT_FILE'
KEY_EXPORT_FILE_BUTTON = 'K_EXPORT_FILE_BUTTON'


def get_route_coordinates(route, euclidean_coords):
    return np.array([euclidean_coords[loc_id] for loc_id in route])


def make_window(window_title: str):
    layout = [
        sg.vtop([
            sg.Frame('Depot', [
                [
                    sg.Column([
                        [sg.Text('Latitude', size=(20, 1)), sg.Text('Longitude', size=(20, 1))],
                        [sg.Input(size=(20, 1), k=KEY_DEPOT_LAT), sg.Input(size=(20, 1), k=KEY_DEPOT_LONG)],
                        [sg.Text('Error: invalid values', k=KEY_DEPOT_ERROR, visible=False, colors='red')]
                    ], pad=(10, 10))
                ]
            ], pad=((10, 20), (10, 10))),
            sg.Frame('Customers', [
                [
                    sg.Column([
                        [sg.Listbox([], k=KEY_CUSTOMERS_INPUT, size=(40, 6))],
                        [sg.Column([
                            [
                                sg.Button('Add', k=KEY_CUSTOMER_ADD, pad=(10, 0)),
                                sg.Button('Edit', k=KEY_CUSTOMER_EDIT, pad=(10, 0)),
                                sg.Button('Delete', k=KEY_CUSTOMER_DELETE, pad=(10, 0)),
                                sg.Input(k=KEY_FILE_IMPORT, visible=False, enable_events=True),
                                sg.FileBrowse('Import CSV', pad=(10, 0),
                                              file_types=(('Comma-separated values', '*.csv'),)),
                                sg.Input(k=KEY_CUSTOMER_EXPORT, visible=False, enable_events=True),
                                sg.FileSaveAs('Export CSV', k=KEY_CUSTOMER_EXPORT_BUTTON, disabled=True, pad=(10, 0),
                                              file_types=(('Comma-separated values', '*.csv'),))
                            ],
                        ], pad=(0, 5))],

                        [
                            sg.Text('Error: ', visible=False, colors='red', k=KEY_CUSTOMERS_ERROR)
                        ]
                    ], pad=(10, 10))
                ]
            ], pad=((20, 10), (10, 10)))
        ]),
        [
            sg.Frame('Parameters', [
                [
                    sg.Column([
                        [sg.Text('Number of big clusters', size=(40, 1)),
                         sg.Text('Minimum customer per subcluster', size=(40, 1))],
                        [sg.Input(k=KEY_NUM_BIG_CLUSTERS, size=(40, 1)),
                         sg.Input(k=KEY_MIN_CUSTOMER_PER_SUBCLUSTER, size=(40, 1))],
                        [
                            sg.Text('Error: invalid values', k=KEY_NUM_BIG_CLUSTERS_ERROR, visible=False, colors='red',
                                    size=(40, 1)),
                            sg.Text('Error: invalid values', k=KEY_MIN_CUSTOMER_PER_SUBCLUSTER_ERROR, visible=False,
                                    colors='red', size=(40, 1))
                        ],
                        [sg.Text('Vehicle capacity', size=(40, 1)), sg.Text('Battery capacity (kWh)', size=(40, 1))],
                        [sg.Input(k=KEY_VEHICLE_CAPACITY, size=(40, 1)),
                         sg.Input(k=KEY_BATTERY_CAPACITY, size=(40, 1))],
                        [
                            sg.Text('Error: invalid values', k=KEY_VEHICLE_CAPACITY_ERROR, visible=False, colors='red',
                                    size=(40, 1)),
                            sg.Text('Error: invalid values', k=KEY_BATTERY_CAPACITY_ERROR, visible=False, colors='red',
                                    size=(40, 1))
                        ],
                        [sg.Text('Battery consumption (kWh/km)', size=(40, 1)),
                         sg.Text('Travel cost (VND/km)', size=(40, 1))],
                        [sg.Input(k=KEY_BATTERY_CONSUMPTION, size=(40, 1)), sg.Input(k=KEY_TRAVEL_COST, size=(40, 1))],
                        [
                            sg.Text('Error: invalid values', k=KEY_BATTERY_CONSUMPTION_ERROR, visible=False,
                                    colors='red',
                                    size=(40, 1)),
                            sg.Text('Error: invalid values', k=KEY_TRAVEL_COST_ERROR, visible=False, colors='red',
                                    size=(40, 1))
                        ],
                        [sg.Text('Operation cost (VND/vehicle)', size=(40, 1)),
                         sg.Text('Recharging cost (VND/station)', size=(40, 1))],
                        [sg.Input(k=KEY_OPERATION_COST, size=(40, 1)), sg.Input(k=KEY_RECHARGING_COST, size=(40, 1))],
                        [
                            sg.Text('Error: invalid values', k=KEY_OPERATION_COST_ERROR, visible=False,
                                    colors='red',
                                    size=(40, 1)),
                            sg.Text('Error: invalid values', k=KEY_RECHARGING_COST_ERROR, visible=False, colors='red',
                                    size=(40, 1)),
                        ]
                    ], pad=(10, 10))
                ]
            ], expand_x=True)
        ],
        [
            sg.Frame('Result', [
                [sg.Column([
                    [sg.Multiline(k=KEY_RESULT, disabled=True, size=(88, 6))],
                    [
                        sg.Column([
                            [
                                sg.Button('Show plot', k=KEY_PLOT, disabled=True, pad=(10, 0)),
                                sg.Input(k=KEY_EXPORT_FILE, visible=False, enable_events=True),
                                sg.FileSaveAs('Export', k=KEY_EXPORT_FILE_BUTTON, disabled=True,
                                              file_types=(('Text files', '*.txt'),), pad=(10, 0))
                            ]
                        ], pad=(0, 5))

                    ]
                ], pad=(10, 10))]
            ], expand_x=True)
        ],
        [
            sg.Push(),
            sg.Button("Optimize", k=KEY_EXECUTE, enable_events=True),
            sg.Push()
        ]
    ]

    window = sg.Window(window_title, layout, font=('default', 12,), metadata=0)
    return window


def get_customer_label(id, lat, long, demand):
    return f'#{id + 1}: ({lat}, {long}) - Demand {demand}'


def get_customer_id(label):
    return int(label.split(':')[0][1:])


def refresh_customers(customers_widget, customers, export_button):
    customers_widget.update(
        [get_customer_label(idx, customer[0], customer[1], customer[2]) for idx, customer in enumerate(customers)])
    disabled = len(customers) == 0
    export_button.update(disabled=disabled)


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def main():
    depot_id = [0]
    depot_long = []
    depot_lat = []
    depot_demand = [0]
    customers = []
    cost = 0
    delivery = []
    travel_cost = None
    operation_cost = None
    recharging_cost = None
    result_txt = ''

    depot_euclidean_coords = []
    customer_euclidean_coords = []
    charging_station_euclidean_coords = []
    depot_color = None
    customer_color = None
    charging_station_color = None

    vns_2_routes = []

    num_big_clusters = None
    min_customer_per_subcluster = None
    vehicle_capacity = None
    battery_capacity = None
    battery_consumption_rate = None

    window = make_window('Route Optimizer')
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == KEY_FILE_IMPORT:
            file_path = values[KEY_FILE_IMPORT]
            f = open(file_path, 'r')
            content = f.read()
            lines = content.split('\n')
            customers = []
            import_error_lines = []
            for idx, line in enumerate(lines):
                data = line.split(',')
                if not line.strip():
                    continue
                if len(data) < 3:
                    import_error_lines.append(idx + 1)
                    continue
                str_lat = data[0]
                str_long = data[1]
                str_demand = data[2]
                if is_float(str_lat) and is_float(str_long) and is_int(str_demand):
                    customers.append([float(str_lat), float(str_long), int(str_demand)])
                else:
                    import_error_lines.append(idx + 1)
            if len(import_error_lines) > 0:
                customers = []
                window[KEY_CUSTOMERS_ERROR].update(
                    f'Import error: invalid data in line(s) {', '.join(map(str, import_error_lines))}', visible=True)
            elif len(customers) == 0:
                window[KEY_CUSTOMERS_ERROR].update('Error: Customer list is empty', visible=True)
            else:
                window[KEY_CUSTOMERS_ERROR].update(visible=False)
            refresh_customers(window[KEY_CUSTOMERS_INPUT], customers, window[KEY_CUSTOMER_EXPORT_BUTTON])
        elif event == KEY_CUSTOMER_ADD:
            added_customer = customer_input_popup()
            if added_customer is not None:
                customers.append(added_customer)
                refresh_customers(window[KEY_CUSTOMERS_INPUT], customers, window[KEY_CUSTOMER_EXPORT_BUTTON])
        elif event == KEY_CUSTOMER_EDIT:
            selected_list = values[KEY_CUSTOMERS_INPUT]
            if len(selected_list) == 0:
                continue
            selected = selected_list[0]
            selected_id = get_customer_id(selected)
            edited_customer = customers[selected_id - 1]
            edited_data = customer_input_popup(selected_id, edited_customer[0], edited_customer[1], edited_customer[2])
            if edited_data is not None:
                customers[selected_id - 1] = edited_data
                refresh_customers(window[KEY_CUSTOMERS_INPUT], customers, window[KEY_CUSTOMER_EXPORT_BUTTON])
        elif event == KEY_CUSTOMER_DELETE:
            selected_list = values[KEY_CUSTOMERS_INPUT]
            if len(selected_list) == 0:
                continue
            selected = selected_list[0]
            selected_id = get_customer_id(selected)
            del customers[selected_id - 1]
            refresh_customers(window[KEY_CUSTOMERS_INPUT], customers, window[KEY_CUSTOMER_EXPORT_BUTTON])
        elif event == KEY_CUSTOMER_EXPORT:
            file_path = values[KEY_CUSTOMER_EXPORT]
            file = open(file_path, 'w')
            for customer in customers:
                file.write(f'{customer[0]}, {customer[1]}, {customer[2]}\n')
            file.close()
        elif event == KEY_PLOT:
            # Initialize the second figure for VNS2
            plt.figure(figsize=(15, 10))

            # Plot depots
            plt.scatter(depot_euclidean_coords[0], depot_euclidean_coords[1], color=depot_color, label='Depot', s=100,
                        marker='o')

            # Plot customers
            plt.scatter(customer_euclidean_coords[0], customer_euclidean_coords[1], color=customer_color,
                        label='Customer', s=50, marker='x')

            # Plot charging stations
            plt.scatter(charging_station_euclidean_coords[0], charging_station_euclidean_coords[1],
                        color=charging_station_color,
                        label='Charging Station', s=95, marker='s')

            for [route, cluster_num, smaller_cluster_num] in vns_2_routes:
                plt.plot(route[:, 0], route[:, 1], linestyle='-', linewidth=1.2, marker='o', markersize=3,
                         label=f'Cluster {cluster_num + 1}.{smaller_cluster_num + 1}')

            plt.xlabel('X (Euclidean)')
            plt.ylabel('Y (Euclidean)')
            plt.title('Routes for All Clusters and Sub-clusters in need of charging')
            plt.legend()
            plt.grid(True)
            plt.show()
        elif event == KEY_EXPORT_FILE:
            file_path = values[KEY_EXPORT_FILE]
            file = open(file_path, 'w')
            file.write(result_txt)
            file.close()
        elif event == KEY_EXECUTE:
            error = False
            result_txt = ''
            cost = 0
            delivery = []
            window[KEY_RESULT].update(result_txt)
            window[KEY_PLOT].update(disabled=True)
            window[KEY_EXPORT_FILE_BUTTON].update(disabled=True)

            try:
                depot_long = [float(values[KEY_DEPOT_LONG])]
                depot_lat = [float(values[KEY_DEPOT_LAT])]
            except ValueError:
                window[KEY_DEPOT_ERROR].update(visible=True)
                error = True
            else:
                window[KEY_DEPOT_ERROR].update(visible=False)

            try:
                num_big_clusters = int(values[KEY_NUM_BIG_CLUSTERS])
                window[KEY_NUM_BIG_CLUSTERS_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_NUM_BIG_CLUSTERS_ERROR].update(visible=True)

            try:
                min_customer_per_subcluster = int(values[KEY_MIN_CUSTOMER_PER_SUBCLUSTER])
                window[KEY_MIN_CUSTOMER_PER_SUBCLUSTER_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_MIN_CUSTOMER_PER_SUBCLUSTER_ERROR].update(visible=True)

            try:
                vehicle_capacity = int(values[KEY_VEHICLE_CAPACITY])
                window[KEY_VEHICLE_CAPACITY_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_VEHICLE_CAPACITY_ERROR].update(visible=True)

            try:
                battery_capacity = int(values[KEY_BATTERY_CAPACITY])
                window[KEY_BATTERY_CAPACITY_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_BATTERY_CAPACITY_ERROR].update(visible=True)

            try:
                battery_consumption_rate = float(values[KEY_BATTERY_CONSUMPTION])
                window[KEY_BATTERY_CONSUMPTION_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_BATTERY_CONSUMPTION_ERROR].update(visible=True)

            try:
                travel_cost = int(values[KEY_TRAVEL_COST])
                window[KEY_TRAVEL_COST_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_TRAVEL_COST_ERROR].update(visible=True)

            try:
                operation_cost = float(values[KEY_OPERATION_COST])
                window[KEY_OPERATION_COST_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_OPERATION_COST_ERROR].update(visible=True)

            try:
                recharging_cost = float(values[KEY_RECHARGING_COST])
                window[KEY_RECHARGING_COST_ERROR].update(visible=False)
            except ValueError:
                error = True
                window[KEY_BATTERY_CONSUMPTION_ERROR].update(visible=True)

            if len(customers) == 0:
                error = True
                window[KEY_CUSTOMERS_ERROR].update('Error: Customer list is empty', visible=True)
            else:
                window[KEY_CUSTOMERS_ERROR].update(visible=False)

            if error:
                continue

            customer_latitudes = [float(customer[0]) for customer in customers]
            customer_longitudes = [float(customer[1]) for customer in customers]
            customer_demands = [customer[2] for customer in customers]

            # KMeans big clusters
            customer_combined_euclidean_coords = np.array(
                [lat_long_to_euclidean(lat, long) for [lat, long, demand] in customers])
            centroids, labels = create_big_clusters(num_big_clusters, customer_combined_euclidean_coords)
            print(centroids, '\n', labels)

            # Convert centroids from Euclidean coordinates back to lat/lon
            centroid_lat_lon = [euclidean_to_lat_long(centroid[0], centroid[1]) for centroid in centroids]

            # Separate latitudes and longitudes into two different arrays
            charging_station_latitudes, charging_station_longitudes = zip(*centroid_lat_lon)
            charging_station_demands = [0] * len(centroids)

            customer_indices = range(1, 1 + len(customers))
            charging_station_indices = range(1 + len(customers), 1 + len(customers) + len(centroids))

            depot_list = []
            customer_list = []
            charging_station_list = []

            populate_location_list(depot_list, depot_id, depot_lat, depot_long, depot_demand)
            populate_location_list(customer_list, customer_indices, customer_latitudes, customer_longitudes,
                                   customer_demands)
            populate_location_list(charging_station_list, charging_station_indices, charging_station_latitudes,
                                   charging_station_longitudes, charging_station_demands)

            # Combine the lists
            vertices = []
            vertex_indices = list(chain(depot_id, customer_indices, charging_station_indices))
            vertex_longitudes = list(chain(depot_long, customer_longitudes, charging_station_longitudes))
            vertex_latitudes = list(chain(depot_lat, customer_latitudes, charging_station_latitudes))
            vertice_demand = list(chain(depot_demand, customer_demands, charging_station_demands))

            # Adding component factors to types of locations
            populate_location_list(vertices, vertex_indices, vertex_latitudes, vertex_longitudes, vertice_demand)

            # Compute distances
            distance_lists = []
            for i in range(len(vertices)):
                distances_from_i = []  # List to hold distances from location i
                for j in range(len(vertices)):
                    dist = haversine_distance(vertices[i]['lat'], vertices[i]['long'], vertices[j]['lat'],
                                              vertices[j]['long'])
                    distances_from_i.append(dist)
                distance_lists.append(distances_from_i)

            # Convert the list of lists into a numpy array (distance matrix)
            distance_matrix = np.array(distance_lists)

            routing_service = RoutingService(distance_matrix, vertices, labels, len(depot_list), len(customer_list),
                                             len(charging_station_list))

            customer_euclidean_coords = lat_long_to_euclidean(customer_latitudes, customer_longitudes)
            depot_euclidean_coords = lat_long_to_euclidean(depot_lat, depot_long)
            charging_station_euclidean_coords = lat_long_to_euclidean(charging_station_latitudes,
                                                                      charging_station_longitudes)
            vertices_euclidean_coords = np.array(
                [lat_long_to_euclidean(lat, long) for lat, long in zip(vertex_latitudes, vertex_longitudes)])

            vns_2_routes = []
            # KMeans smaller clusters
            for cluster_num in range(num_big_clusters):
                result_txt += f'Cluster {cluster_num + 1}:\n'
                cluster_customers_idx = np.where(labels == cluster_num)[0]
                cluster_customers_coords = customer_combined_euclidean_coords[cluster_customers_idx]
                cluster_demands = [customer_demands[i] for i in cluster_customers_idx]

                smaller_clusters, demand_sums = create_smaller_clusters(cluster_customers_coords, cluster_demands,
                                                                        vehicle_capacity, min_customer_per_subcluster)

                for smaller_cluster_num in range(max(smaller_clusters) + 1):
                    result_txt += f'Subcluster {cluster_num + 1}.{smaller_cluster_num + 1} (Total demand {demand_sums[smaller_cluster_num]}):\n'
                    smaller_cluster_customers = np.where(smaller_clusters == smaller_cluster_num)[0]
                    for idx in smaller_cluster_customers:
                        customer_idx = cluster_customers_idx[idx]
                        result_txt += f'\t- Customer {customer_idx + 1}: at ({customers[customer_idx][0]}, {customers[customer_idx][1]}) with demand {cluster_demands[idx]}\n'

                    result_txt += '\n'

                    smaller_cluster_customers_indices = [cluster_customers_idx[idx] + 1 for idx in
                                                         smaller_cluster_customers]
                    route_customers = [0] + smaller_cluster_customers_indices

                    # Apply VNS 1 to the smaller cluster to get the initial route
                    optimal_route_vns_1, travel_vns_1 = routing_service.vns_1(route_customers)

                    # Apply VNS 2 to the smaller cluster in need of charging
                    optimal_route_vns_2, travel_vns_2, optimal_route_txt_vns_2 = routing_service.vns_2(optimal_route_vns_1, battery_capacity,
                                                                              battery_consumption_rate)

                    delivery.append((cluster_num, smaller_cluster_num, optimal_route_vns_2, optimal_route_txt_vns_2))

                    route_line_vns_2 = get_route_coordinates(optimal_route_vns_2, vertices_euclidean_coords)

                    vns_2_routes.append([route_line_vns_2, cluster_num, smaller_cluster_num])

            max_route_length = max(len(route) for a, b, route, route_txt in delivery)
            delivery_array = np.zeros((len(delivery), max_route_length), dtype=int)
            for i, (cluster_num, smaller_cluster_num, route, route_txt) in enumerate(delivery):
                delivery_array[i, :len(route)] = route

            total_distance_sum = 0
            result_txt += '\n'
            for i, (cluster_num, smaller_cluster_num, route, route_txt) in enumerate(delivery):
                a = delivery_array[i, :]  # Fetch the route from the delivery_array
                distance = routing_service.kc_2(a)
                total_distance_sum += distance
                result_txt += f'Electric vehicle {i + 1} in Cluster {cluster_num + 1}.{smaller_cluster_num + 1}, Distance: {distance:.4f}km\n'
                result_txt += 'Route taken:\n'
                result_txt += route_txt
                result_txt += '\n'

            result_txt += f'\nTotal distance traveled by all vehicles: {total_distance_sum:.4f}km\n'
            num_route = 0

            for i, route in enumerate(delivery_array):
                num_route += 1

            result_txt += f'\nNumber of electric vehicles : {num_route}\n'
            result_txt += f'\nNumber of charging stations : {len(charging_station_indices)}\n'

            cost += (total_distance_sum * travel_cost +
                     num_route * operation_cost +
                     recharging_cost * len(charging_station_indices))

            result_txt += f'\nTotal operating cost per day : {cost:,.2f}VND\n'
            window[KEY_RESULT].update(result_txt)

            depot_color = 'red'
            customer_color = 'blue'
            charging_station_color = 'green'

            window[KEY_PLOT].update(disabled=False)
            window[KEY_EXPORT_FILE_BUTTON].update(disabled=False)
    window.close()


if __name__ == '__main__':
    main()
