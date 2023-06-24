import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data=np.random.rand(86,20)
data1=np.random.rand(86,20)
data2=np.random.rand(86,20)
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(data2,Z=data,Y=data1,cmap='jet')
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
