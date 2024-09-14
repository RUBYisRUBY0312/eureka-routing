# Generate an initial solution using a greedy nearest neighbor approach
import random
import numpy as np
from itertools import chain


def _two_opt(route):
    new_route = route[:]
    id_1, id_2 = random.sample(range(1, len(route) - 1), 2)  # Don't swap depot (index 0 and last)
    new_route[id_1], new_route[id_2] = new_route[id_2], new_route[id_1]
    return new_route


class RoutingService:
    def __init__(self, distance_matrix, vertices, cluster_labels, n_depot, n_customer, n_charging_station):
        self._distance_matrix = distance_matrix
        self._cluster_labels = cluster_labels
        self._vertices = vertices
        # Define indices for each location type
        self._depot_indices = range(n_depot)
        self._customer_indices = range(n_depot, n_depot + n_customer)
        self._charging_station_indices = range(n_depot + n_customer, n_depot + n_customer + n_charging_station)
        self._vertex_indices = list(chain(self._depot_indices, self._customer_indices, self._charging_station_indices))

        # Extract distance matrices
        self._depot_to_customers_matrix = distance_matrix[self._depot_indices][:, self._customer_indices]
        self._customers_to_customers_matrix = distance_matrix[self._customer_indices][:, self._customer_indices]
        self._customers_to_charging_stations_matrix = distance_matrix[self._customer_indices][:,
                                               self._charging_station_indices]
        self._charging_stations_to_customers_matrix = distance_matrix[self._charging_station_indices][:,
                                               self._customer_indices]
        self._customers_to_depot_matrix = distance_matrix[self._customer_indices][:, self._depot_indices]
        self._charging_stations_to_depot_matrix = distance_matrix[self._charging_station_indices][:, self._depot_indices]

        self._depot_to_customers = np.zeros((n_depot, n_customer))
        self._customers_to_customers = np.zeros((n_customer, n_customer))
        self._customers_to_charging_stations = np.zeros((n_customer, n_charging_station))
        self._charging_stations_to_customers = np.zeros((n_charging_station, n_customer))
        self._charging_stations_to_depot = np.zeros((n_charging_station, n_depot))
        self._customers_to_depot = np.zeros((n_customer, n_depot))

        for i in self._customer_indices:
            self._depot_to_customers[:, i - self._customer_indices[0]] = self._dist_depot_to_customer(i)
            self._customers_to_customers[i - self._customer_indices[0], :] = self._dist_customer_to_customer(i)
            self._customers_to_charging_stations[i - self._customer_indices[0], :] = self._dist_customer_to_charging_station(i)
            self._customers_to_depot[i - self._customer_indices[0], :] = self._dist_customer_to_depot(i)

        for i in self._charging_station_indices:
            self._charging_stations_to_customers[i - self._charging_station_indices[0], :] = self._dist_charging_station_to_customer(i)
            self._charging_stations_to_depot[i - self._charging_station_indices[0], :] = self._dist_charging_station_to_depot(i)


    def _dist_depot_to_customer(self, customer_index):
        if customer_index not in self._customer_indices:
            raise ValueError(f"Customer index {customer_index} is out of range")
        col_index = customer_index - 1  # Adjust for 0-based indexing
        output = self._depot_to_customers_matrix[0, col_index]
        return output

    def _depot_to_routed_customer(self, customer_index_to):
        output = self._depot_to_customers[0, customer_index_to - self._customer_indices[0]]
        return output

    def _dist_customer_to_customer(self, customer_index):
        if customer_index not in self._customer_indices:
            raise ValueError(f"Customer index {customer_index} is out of range")
        row_index = customer_index - 1  # Adjust for 0-based indexing
        output = self._customers_to_customers_matrix[row_index]
        return output

    def _customer_to_routed_customer(self, customer_index_from, customer_index_to):
        output = self._customers_to_customers[customer_index_from - self._customer_indices[0], customer_index_to - self._customer_indices[0]]
        return output

    def _dist_customer_to_charging_station(self, customer_index):
        if customer_index not in self._customer_indices:
            raise ValueError(f"Customer index {customer_index} is out of range")
        # Find the row index of the specific customer
        row_index = customer_index - 1  # Adjust for 0-based indexing
        output = self._customers_to_charging_stations_matrix[row_index]
        return output

    def _customer_to_assigned_charging_station(self, customer_index):
        if customer_index not in self._customer_indices:
            raise ValueError(f"Customer index {customer_index} is out of range")
        # Find the cluster assigned to the customer
        cluster = self._cluster_labels[customer_index - 1]
        output = self._customers_to_charging_stations[customer_index - 1, cluster]
        return output

    def _dist_charging_station_to_customer(self, charging_station_index):
        if charging_station_index not in self._charging_station_indices:
            raise ValueError(f"Charging station index {charging_station_index} is out of range")
        row_index = charging_station_index - self._charging_station_indices[
            0]  # Find the row index of the specific charging station
        output = self._charging_stations_to_customers_matrix[row_index]
        return output

    def _charging_station_to_assigned_customer(self, charging_station_index, customer_index):
        a = self._charging_station_indices
        b = self._customer_indices
        loc_id_to_row_index = {loc_id: index for index, loc_id in
                               enumerate(idx for idx in self._vertex_indices if idx in a)}
        row_index = loc_id_to_row_index[charging_station_index]
        loc_id_to_col_index = {loc_id: index for index, loc_id in
                               enumerate(idx for idx in self._vertex_indices if idx in b)}
        col_index = loc_id_to_col_index[customer_index]
        output = self._charging_stations_to_customers[row_index, col_index]
        return output

    def _dist_charging_station_to_depot(self, charging_station_index):
        if charging_station_index not in self._charging_station_indices:
            raise ValueError(f"Charging station index {charging_station_index} is out of range")
        row_index = charging_station_index - self._charging_station_indices[0]
        output = self._charging_stations_to_depot_matrix[row_index]
        return output

    def _charging_station_to_routed_depot(self, charging_station_index):
        a = self._charging_station_indices
        loc_id_to_row_index = {loc_id: index for index, loc_id in
                               enumerate(idx for idx in self._vertex_indices if idx in a)}
        row_index = loc_id_to_row_index[charging_station_index]
        col_index = 0
        output = self._charging_stations_to_depot[row_index, col_index]
        return output

    def _dist_customer_to_depot(self, customer_index):
        if customer_index not in self._customer_indices:
            raise ValueError(f"Charging station index {customer_index} is out of range")
        row_index = customer_index - 1
        output = self._customers_to_depot_matrix[row_index, 0]
        return output

    def _initial_route(self, customers):
        route = [0]  # Start from the depot
        remaining_customers = customers[1:]  # Exclude depot from customers list

        current_location = 0  # Start from depot
        while remaining_customers:
            if current_location != 0:
                next_customer = min(remaining_customers, key=lambda cus: self._customer_to_routed_customer(current_location, cus))
            else:
                next_customer = min(remaining_customers, key=lambda cus: self._depot_to_routed_customer(cus))
            route.append(next_customer)
            remaining_customers.remove(next_customer)
            current_location = next_customer

        route.append(0)  # Return to the depot
        return route

    # Function to calculate the total distance of a route
    def _total_route_distance(self, route):
        total_distance = 0

        # Start at the depot, then visit next customer
        cus_id_to = route[1]
        total_distance += self._depot_to_routed_customer(cus_id_to)  # From depot to first customer

        # From latest customer, continue its journey
        for i in range(len(route) - 1):
            cus_id_1 = route[i]
            cus_id_2 = route[i + 1]
            total_distance += self._customer_to_routed_customer(cus_id_1, cus_id_2)  # Between customers

        # From last customer back to the depot
        cus_id_from = route[-1]
        total_distance += self._cus_to_routed_dep(cus_id_from)  # From last customer back to depot

        return total_distance

    # Swap two customers in the route (a neighborhood operator)

    def _cus_to_routed_dep(self, cus_index_from):
        output = self._customers_to_depot[cus_index_from - self._customer_indices[0], 0]
        return output

    # Apply Variable Neighborhood Search (VNS) algorithm
    def vns_1(self, customers, max_iterations=100):
        # Step 1: Generate an initial solution
        current_route = self._initial_route(customers)
        current_distance = self._total_route_distance(current_route)

        # Step 2: Explore a new neighborhood by swapping two customers
        iteration = 0
        while iteration < max_iterations:
            # 2.1 Shaking: Apply random neighborhood structure (e.g., 2-opt swap)
            new_route = _two_opt(current_route)
            # Local Search: Check the new route's total distance
            new_distance = self._total_route_distance(new_route)

            # 2.2 Move or Not: If the new route is better, accept it
            if new_distance < current_distance:
                current_route = new_route
                current_distance = new_distance

            iteration += 1  # Move to the next iteration if no improvement

        return current_route, current_distance

    # Function to print the battery state along the route
    def vns_2(self, route, battery_capacity, battery_consumption_rate):
        route_txt = ''
        last_position = self._vertices[route[0]]['id']  # Start at the depot
        max_battery_distance = battery_capacity / battery_consumption_rate  # Maximum distance the battery can cover
        battery_state = max_battery_distance  # Initial battery state in km
        travel = 0
        route_txt += f'\t- Starting at depot with battery capacity: {battery_state:.2f}km\n'

        i = 1
        while i < len(route) - 1:
            current_position = self._vertices[route[i]]['id']
            distance = self._distance_matrix[last_position, current_position]
            battery_state -= distance  # Directly subtract the distance traveled from the battery
            travel += distance

            next_position = self._vertices[route[route.index(current_position) + 1]]['id']
            station = self._cluster_labels[next_position - 1] + len(self._customer_indices) + len(self._depot_indices)
            distance_2 = self._distance_matrix[station, next_position]

            route_txt += f'\t- Travelled to customer {route[i]} with remaining battery: {battery_state:.2f}km\n'

            # Check if battery state is within the recharge range
            if battery_state <= distance_2:
                # Find nearest charging station dynamically
                nearest_station = self._cluster_labels[current_position - 1] + len(self._customer_indices) + len(
                    self._depot_indices)
                distance_to_station = self._customer_to_assigned_charging_station(current_position)
                route_txt += f'\t- From customer {current_position} visited charging station {nearest_station} with distance: {distance_to_station:.4f}km. Battery recharged to {max_battery_distance:.2f}km\n'

                battery_state = max_battery_distance  # Recharge battery to full capacity

                # Insert the charging station index into the route
                route.insert(i + 1, nearest_station)
                i += 1  # Skip to the next customer after the charging station

                # Assume the route from the charging station continues directly
                last_position = nearest_station
            else:
                last_position = current_position

            i += 1

        # Return to the depot
        return_to_depot_distance = self._distance_matrix[last_position, self._vertices[route[-1]]['id']]
        battery_state -= return_to_depot_distance
        route_txt += f'\t- Travelled back to depot with remaining battery: {battery_state:.2f}km\n'

        return route, travel, route_txt  # Return the route array as it was passed in

    def kc_2(self, b):
        total_distance = 0
        for i in range(0, len(b) - 1):
            total_distance += self._distance_matrix[b[i], b[i + 1]]
        return total_distance