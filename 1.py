import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

car_data = pd.read_csv("Toyota.csv", index_col=0, na_values=["??", "????"])
car_data.dropna(axis=0, inplace=True)
car_data
plt.figure(figsize=(10, 10))
ax = plt.axes(projection="3d")
x = car_data["Age"]
y = car_data["KM"]
xx, yy = np.meshgrid(x, y)
z = np.array(car_data["Price"])
ax.set_zlabel("Price")
ax.plot_surface(xx, yy, z.reshape(-1, 1), cmap="Purples")
plt.title("Toyota surface plot")
plt.xlabel("Age")
plt.ylabel("KM")
plt.show()

#---------------------------------------------

#best first search
import heapq

def bfs(start,end,graph):
  frontier=[(0,start)]
  explore=set()

  while frontier:
    (curr_cost,curr)=heapq.heappop(frontier)

    if curr==end:
      return curr_cost

    explore.add(curr)

    for node,cost in graph[curr]:
      if node not in [i[0] for i in frontier] and node not in explore:
        heapq.heappush(frontier,(curr_cost+cost,node))

        print(f'Added {node} with cost {cost}')


  return None

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
    'P': [],
    'Q':[]
}

res=bfs('A','C',graph)

print(res)
