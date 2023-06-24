import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# create some sample data
np.random.seed(1)
n = 100
data = np.random.randn(n, 4)
# create a heat map using Seaborn
sns.heatmap(data)
plt.show()

#-----------------------------------

class Tree:
    def __init__(self, value, children=[]):
        self.value = value
        self.children = children

    def __repr__(self):
        return 'Tree({0},{1})'.format(self.value, self.children)

# Define the game tree using the Tree data structure
game_tree = Tree(0, [
    Tree(0, [
        Tree(3),
        Tree(12)
    ]),
    Tree(0, [
        Tree(8),
        Tree(2)
    ])
])

# Define the Minimax algorithm function with solution path
def minimax(node, depth, maximizing_player):
    # Check if the node is a leaf or if the maximum depth has been reached
    if depth == 0 or not node.children:
        return node.value, [node.value]
    # Apply the Minimax algorithm
    if maximizing_player:
        max_value = float("-inf")
        max_path = []
        for child_node in node.children:
            child_value, child_path = minimax(child_node, depth - 1, False)
            if child_value > max_value:
                max_value = child_value
                max_path = [node.value] + child_path
        return max_value, max_path
    else:
        min_value = float("inf")
        min_path = []
        for child_node in node.children:
            child_value, child_path = minimax(child_node, depth - 1, True)
            if child_value < min_value:
                min_value = child_value
                min_path = [node.value] + child_path
        return min_value, min_path

# Example usage:
optimal_value, optimal_path = minimax(game_tree, 2, True)
print("Optimal value:", optimal_value)
print("Optimal path:", optimal_path)
