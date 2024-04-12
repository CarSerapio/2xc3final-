#!/usr/bin/env python
# coding: utf-8

# ## Part 5
# 
# Organize you code as per the below Unified Modelling Language (UML) diagram in Figure 2. Furthermore, consider the points listed below and discuss these points in a section labelled Part 4 in your report (where appropriate). 
# 
# * Instead of re-writing A* algorithm for this part, treat the class from UML as an “adapter”. 
# * Discuss what design principles and patterns are being used in the diagram
# * The UML is limited in the sense that graph nodes are represented by the integers. How would you alter the UML diagram to accommodate various needs such as nodes being represented Strings or carrying more information than their names.? Explain how you would change the design in Figure 2 to be robust to these potential changes.
# * Discuss what other types of graphs we could have implement “Graph”. What other implementations exist?

# This is just a test input graph to test with the ShortPathFinder class and the corresponding SP algorithms.

# In[1]:


from abc import ABC, abstractmethod


# In[2]:


class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)
    
    def get_nodes(self,):
        return list(self.adj.keys())
    def heuristic(self, node, destination):
        if node in self.adj.keys():
            if destination in self.adj[node]:
                return self.w(node, destination)
            else:
                total_weight = 0
                edge_count = 0
                for neighbors in self.adj[node]:
                        edge_weight = self.w(node, neighbors)
                        total_weight += edge_weight
                        edge_count += 1
                if edge_count == 0:
                    return 0  # Return 0 if there are no edges
                return total_weight / edge_count


# This is needed for the Dijkstra's shortest path algorithms.

# In[3]:


class Item:
    def __init__(self, value, key):
        self.value = value
        self.key = key

class MinHeap:
    def __init__(self, elements):
        self.heap = elements
        self.positions = {element.value: i for i, element in enumerate(elements)}
        self.size = len(elements)
        self.build_heap()

    def parent(self, i):
        return (i - 1) // 2

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        self.positions[self.heap[i].value], self.positions[self.heap[j].value] = i, j

    def min_heapify(self, i):
        l = self.left(i)
        r = self.right(i)
        smallest = i
        if l < self.size and self.heap[l].key < self.heap[i].key:
            smallest = l
        if r < self.size and self.heap[r].key < self.heap[smallest].key:
            smallest = r
        if smallest != i:
            self.swap(i, smallest)
            self.min_heapify(smallest)

    def build_heap(self):
        for i in range(self.size // 2, -1, -1):
            self.min_heapify(i)

    def extract_min(self):
        min_element = self.heap[0]
        self.size -= 1
        self.heap[0] = self.heap[self.size]
        self.positions[self.heap[0].value] = 0
        self.heap.pop()
        self.min_heapify(0)
        return min_element

    def decrease_key(self, value, new_key):
        i = self.positions[value]
        if new_key < self.heap[i].key:
            self.heap[i].key = new_key
            while i > 0 and self.heap[self.parent(i)].key > self.heap[i].key:
                self.swap(i, self.parent(i))
                i = self.parent(i)

    def insert(self, element):
        self.size += 1
        self.heap.append(element)
        self.positions[element.value] = self.size - 1
        self.decrease_key(element.value, element.key)

    def is_empty(self):
        return self.size == 0


# UML diagram implementation starts here: 

# In[4]:


class ShortPathFinder:

    def __init__(self):
        self.Graph = Graph()
        #self.Graph = DirectedWeightedGraph()
        self.SPAlgorithm = SPAlgorithm()

    def calc_short_path(self, source, dest):
        dist = self.SPAlgorithm(self.Graph, source, dest)
        total = 0
        for key in dist.keys():
            total += dist[key]
        return total

    @property
    def set_graph(self):
        self._Graph

    @set_graph.setter
    def set_graph(self, graph):
        self._Graph = graph

    @property
    def set_algorithm(self,):
        self._SPAlgorithm
    
    @set_algorithm.setter
    def set_algorithm(self, algorithm):
        self._SPAlgorithm = algorithm


# In[5]:


class SPAlgorithm():

    @abstractmethod
    def calc_sp(self, g: DirectedWeightedGraph, source: int, k: int):
        pass


# In[6]:


class Dijkstra(SPAlgorithm): 
    
    def calc_sp(self, g, source):
        paths = {source: [source]} 
        dist = {} 
        nodes = list(g.adj.keys())
        
        Q = MinHeap([])

        for node in nodes:
            Q.insert(Item(node, float("inf")))
            dist[node] = float("inf")
        
        Q.decrease_key(source, 0)
    
        while not Q.is_empty(): 
            current_element = Q.extract_min() 
            current_node = current_element.value
            dist[current_node] = current_element.key 
            for neighbour in g.adj[current_node]:
                if dist[current_node] + g.w(current_node, neighbour) < dist[neighbour]:
                    Q.decrease_key(neighbour, dist[current_node] + g.w(current_node, neighbour))
                    dist[neighbour] = dist[current_node] + g.w(current_node, neighbour)
                    paths[neighbour] = paths.get(current_node, []) + [neighbour]
                
        return dist


# In[7]:


class Bellman_Ford(SPAlgorithm): 
    
    def calc_sp(self, g, source): 
        paths = {source: [source]} 
        dist = {} 
        nodes = list(g.adj.keys())

        for node in nodes: 
            dist[node] = float("inf")
        
        dist[source] = 0 
    
        for _ in range(g.number_of_nodes() - 1):
            for node in nodes: 
                for neighbour in g.adj[node]: 
                    if dist[neighbour] > dist[node] + g.w(node,neighbour):
                        dist[neighbour] = dist[node] + g.w(node,neighbour) 
                        paths[neighbour] = paths.get(node, []) + [neighbour]
                        
        for node in nodes:
            for neighbour in g.adj[node]:
                if dist[neighbour] > dist[node] + g.w(node, neighbour):
                    # negative cycle detected, return infinity
                    return float('inf'), relax_count, paths
                                        
        return dist


# In[8]:


class AStar(SPAlgorithm): 
    def calc_sp(self, graph, source, destination, h):
    
        openset = MinHeap([])
        openset.insert(Item(source, float('inf')))
        cameFrom = {}
        gscore = {node : float('inf') for node in graph.adj.keys()}
        gscore[source] = 0
        fscore = {node : float('inf') for node in graph.adj.keys()}
        fscore[source] = h[source]
        while not openset.is_empty():
            current_min = openset.extract_min()        
            if current_min == destination:
                break
            if current_min is None:
                break
            current = current_min.value
            for neighbour in graph.adj[current]:
                t_gscore = gscore[current]+ graph.w(current, neighbour)
                if t_gscore < gscore[neighbour]:
                    cameFrom[neighbour] = current
                    gscore[neighbour] = t_gscore
                    fscore[neighbour] = t_gscore + h[neighbour]
                    if neighbour not in openset.positions:
                        openset.insert(Item(neighbour, fscore[neighbour]))
        shortestpath = []
        current = destination
        while current in cameFrom:
            shortestpath.insert(0, current)
            current = cameFrom[current]
        shortestpath.insert(0, source)
        return shortestpath


# In[9]:


class Graph(ABC):

    def __init__(self):
        self.adj = {}
        self.weights = {}

    @abstractmethod
    def adjacent_nodes(self, node):
        pass

    @abstractmethod
    def add_node(self, node):
        pass

    @abstractmethod
    def add_edge(self, node1, node2, weight):
        pass

    @abstractmethod
    def number_of_nodes(self):
        pass
    
    @abstractmethod
    def w(self, node):
        pass


# In[10]:


class WeightedGraph(Graph):

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]
        
class HeuristicGraph(WeightedGraph): 
    
    def __init__(self, heuristic): 
        self.heuristic = heuristic 
        
    @property
    def get_heuristic(self): 
        return self.heuristic 


# Test cases: 

# In[13]:


# Create an instance of DirectedWeightedGraph
g = DirectedWeightedGraph()

# Add nodes
g.add_node(0)
g.add_node(1)
g.add_node(2)
g.add_node(3)

# Add edges with weights
g.add_edge(0, 1, 4)
g.add_edge(0, 2, 3)
g.add_edge(0, 3, 10)
g.add_edge(1, 2, 1)
g.add_edge(1, 3, 2)
g.add_edge(2, 3, 5)

source_node = 0 
dest_node = 3 
relaxation_limit = 100 # Arbitrarily high value to get full shortest paths 


# In[14]:


finder1 = Dijkstra()
finder2 = Bellman_Ford() 
finder3 = AStar() 

# Calculate the shortest path
shortest_distances = finder1.calc_sp(g, source_node)

print(shortest_distances)

shortest_distances = finder2.calc_sp(g, source_node)

print(shortest_distances)

h = {node : float(g.heuristic(node, dest_node)) for node in g.adj.keys()}
shortest_distances = finder3.calc_sp(g, source_node, dest_node, h)

print(shortest_distances)


# In[ ]:




