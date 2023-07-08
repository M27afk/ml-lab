#Visualize the n-dimensional data using contour plots.
#  imports
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0, 50 ,3)
y = np.arange(0, 50, 4)

X, Y = np.meshgrid(x, y)
Z = np.sin(X/2) +  np.cos(Y/4)
  

plt.contour(X, Y, Z, cmap='viridis');
#-------------------------------------------

import heapq

def a_star(graph, start, goal, heuristic):
    # Initialize the priority queue with the start node
    frontier = [(0 + heuristic[start], start)]
    # Initialize the cost dictionary with the start node
    cost = {start: 0}
    # Initialize the parent dictionary with the start node
    parent = {start: None}
    # Initialize the explored set
    explored = set()

    # Loop until the frontier is empty
    while frontier:
        # Pop the node with the lowest f-score
        (f_score, current_node) = heapq.heappop(frontier)

        # Check if the current node is the goal
        if current_node == goal:
            # Reconstruct the path from the goal to the start
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = parent[current_node]
            path.reverse()
            return (path, cost[goal])

        # Add the current node to the explored set
        explored.add(current_node)

        # Explore the neighbors of the current node
        for neighbor, neighbor_cost in graph[current_node]:
            # Calculate the tentative g-score
            tentative_g_score = cost[current_node] + neighbor_cost
            # Check if the neighbor is already in the explored set
            if neighbor in explored:
                # If the tentative g-score is higher than the current g-score, skip this neighbor
                if tentative_g_score >= cost.get(neighbor, float('inf')):
                    continue

            # Check if the neighbor is not in the frontier or the tentative g-score is lower than the current g-score
            if neighbor not in [node[1] for node in frontier] or tentative_g_score < cost.get(neighbor, float('inf')):
                # Update the cost and parent dictionaries
                cost[neighbor] = tentative_g_score
                parent[neighbor] = current_node
                # Add the neighbor to the frontier with its priority being its f-score
                heapq.heappush(frontier, (tentative_g_score + heuristic[neighbor], neighbor))

    # If the goal cannot be reached, return None
    return None

# Example graph
graph = {
    'A': [('B', 5), ('C', 6)],
    'B': [('D', 4), ('E', 7)],
    'C': [('F', 9), ('G', 8)],
    'D': [('H', 3)],
    'E': [('I', 6)],
    'F': [('J', 5)],
    'G': [('K', 7)],
    'H': [('L', 1)],
    'I': [('M', 2)],
    'J': [('N', 3)],
    'K': [('O', 4)],
    'L': [],
    'M': [],
    'N': [],
    'O': [('P', 1)],
    'P': []
}

# Heuristic function
heuristic = {
    'A': 10,
    'B': 8,
    'C': 7,
    'D': 6,
    'E': 8,
    'F': 3,
    'G': 2,
    'H': 5,
    'I': 6,
    'J': 3,
    'K': 2,
    'L': 1,
    'M': 4,
    'N': 2,
    'O': 4,
    'P': 0
}

# Get start and goal nodes from the user
start = input("Enter the start node: ")
goal = input("Enter the goal node: ")

# Run the Best First Search algorithm
result = a_star(graph, start, goal,heuristic)

# Print the result
if result is not None:
    print(f"The minimum cost from {start} to {goal} is {result}.")
else:
    print(f"There is no path from {start} to {goal}.")
