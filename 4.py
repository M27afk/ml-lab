import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create some sample data
np.random.seed(1)
n = 100
data = np.random.randn(n, 4)

# create a box plot for each dimension
for i in range(4):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data[:, i])
    plt.title('Dimension {}'.format(i + 1))
# show the figures

#data = test.iloc[:,3].values
#sns.boxplot(np.asarray([data]))
#plt.xlabel("Age")

plt.show()

#---------------------------



class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.alpha = float('-inf')
        self.beta = float('inf')

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return not self.children

def min_max_with_ab_pruning(node, depth, alpha, beta, maximizing_player):
    if depth == 0 or node.is_leaf():
        return node.value

    if maximizing_player:
        max_eval = float('-inf')
        best_child = None

        for child in node.children:
            child_eval = min_max_with_ab_pruning(child, depth-1, alpha, beta, False)
            if child_eval > max_eval:
                max_eval = child_eval
                best_child = child

            alpha = max(alpha, max_eval)
            if alpha >= beta:
                break

        node.alpha = max_eval
        if node.alpha >= node.beta:
            pruned_nodes.append(node)

        return max_eval

    else:
        min_eval = float('inf')
        best_child = None

        for child in node.children:
            child_eval = min_max_with_ab_pruning(child, depth-1, alpha, beta, True)
            if child_eval < min_eval:
                min_eval = child_eval
                best_child = child

            beta = min(beta, min_eval)
            if alpha >= beta:
                break

        node.beta = min_eval
        if node.alpha >= node.beta:
            pruned_nodes.append(node)

        return min_eval
root = Node(0)
node_b = Node(3)
node_c = Node(-3)
node_d = Node(2)
node_e = Node(1)
node_f = Node(-2)
node_g = Node(5)
node_h = Node(-5)

root.add_child(node_b)
root.add_child(node_c)
node_b.add_child(node_d)
node_b.add_child(node_e)
node_b.add_child(node_f)
node_d.add_child(node_g)
node_d.add_child(node_h)

# Initialize variables
pruned_nodes = []
maximizing_player = True

# Run the algorithm with alpha-beta pruning and get the optimal value and path
optimal_value = min_max_with_ab_pruning(root, 5, float('-inf'), float('inf'), maximizing_player)

# Print the optimal value and path
print("Optimal value:", optimal_value)
path = [root]
current_node = root
while current_node.children:
    if maximizing_player:
        current_node = max(current_node.children, key=lambda x: x.alpha)
        maximizing_player = False
    else:
        current_node = min(current_node.children, key=lambda x: x.beta)
        maximizing_player = True
    path.append(current_node)

print("Solution path:")
for node in path:
    print(node.value)

# Print the pruned nodes
if pruned_nodes:
    print("Pruned nodes:")
    for node in pruned_nodes:
        print(node.value)
else:
    print("No nodes were pruned.")
     
