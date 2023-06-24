import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

# create some sample data
np.random.seed(1)
n = 100
data = np.random.randn(n, 4)

# create a box plot for each dimension
for i in range(4):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data[:, i])
    plt.title('Dimension {}'.format(i + 1))

# create a figure for plotly
fig = go.Figure()
for i in range(4):
    fig.add_trace(go.Box(x=data[:, i]))

# show the figures
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
