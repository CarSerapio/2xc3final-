#!/usr/bin/env python
# coding: utf-8

# ## Part 2
# Dijkstra’s and Bellman Ford’s are single source shortest path algorithms. However, many times we are
# faced with problems that require us to solve shortest path between all pairs. This means that the algorithm
# needs to find the shortest path from every possible source to every possible destination. For every pair of
# vertices u and v, we want to compute shortest path 𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒(𝑢, 𝑣) and the second-to-last vertex on the
# shortest path 𝑝𝑟𝑒𝑣𝑖𝑜𝑢𝑠(𝑢, 𝑣). How would you design an all-pair shortest path algorithm for both positive
# edge weights and negative edge weights? Implement a function that can address this.
# 

# In[9]:


import random


# In[21]:


# weighted digraph 
class DWG:

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

    def num_nodes(self):
        return len(self.adj)
    def print_graph(self):
        print("Directed Graph Content:")
        for node in self.adj:
            print(f"Node {node}:")
            for neighbour in self.adj[node]:
                weight = self.weights.get((node, neighbour), "N/A")
                print(f"  -> Neighbour: {neighbour}, Weight: {weight}")


# In[22]:


# min pq necessary for djisktra's 
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


# In[169]:


def dijkstra_all_pairs(graph):
    paths = {}
    prev = {}
    for src in graph.adj.keys():
        dist = {}
        previous = {}
        heap_elements = []
        for vertex in graph.adj.keys():
            heap_elements.append(Item(vertex, float('inf')))
        Q = MinHeap(heap_elements)
        Q.decrease_key(src, 0)
        while not Q.is_empty():
            current_element = Q.extract_min()
            u = current_element.value
            dist[u] = current_element.key
            for v in graph.adjacent_nodes(u):
                weight = graph.w(u, v)
                if weight is not None:
                    if dist[u] + weight < dist.get(v, float('inf')):
                        Q.decrease_key(v, dist[u] + weight)
                        dist[v] = dist[u] + weight
                        previous[v] = u  # Update the predecessor for node v
        paths[src] = {}
        for vertex in graph.adj.keys():
            if vertex not in dist:
                paths[src][vertex] = float('inf')
            else:
                paths[src][vertex] = dist[vertex]
        prev[src] = {}
        for vertex in graph.adj.keys():
            if vertex in previous:
                prev[src][vertex] = previous[vertex]
            else:
                prev[src][vertex] = None
                print("Paths:")
        print(f"Source: {src}")
        for vertex, distance in paths[src].items():
            print(f" - Vertex: {vertex}, Distance: {distance}")
        print("\nPrevious Nodes:")
        print(f"Source: {src}")
        for vertex, prev_node in prev[src].items():
            print(f" - Vertex: {vertex}, Previous Node: {prev_node}")
        print()
    return paths, prev


# In[170]:


def bellman_ford_all_pairs(g):
    paths = {}
    prev = {}
    for src in g.adj.keys():
        dist = {}
        distance = {src: [src]}
        nodes = list(g.adj.keys())
        relax_count = {}
        previous = {}

        for i in g.adj.keys():
            relax_count[i] = 0

        for node in nodes:
            dist[node] = float("inf")

        dist[src] = 0

        for _ in range(g.num_nodes() - 1):
            for node in nodes:
                for neighbour in g.adj[node]:
                    if relax_count[neighbour] < g.num_nodes() - 1 and dist[neighbour] > dist[node] + g.w(node, neighbour):
                        dist[neighbour] = dist[node] + g.w(node, neighbour)
                        previous[neighbour] = node
                        relax_count[neighbour] += 1
        paths[src] = {}
        prev[src] = {}
        for vertex in g.adj.keys():
            paths[src][vertex] = dist[vertex]
            if vertex in previous:
                prev[src][vertex] = previous[vertex]
            else:
                prev[src][vertex] = None
        print("Paths:")
        print(f"Source: {src}")
        for vertex, distance in paths[src].items():
            print(f" - Vertex: {vertex}, Distance: {distance}")
        print("\nPrevious Nodes:")
        print(f"Source: {src}")
        for vertex, prev_node in prev[src].items():
            print(f" - Vertex: {vertex}, Previous Node: {prev_node}")
        print()
    return paths, prev


# In[171]:


def all_pairs_distance(g):
    graph_weights = False
    for u in g.adj:
        for v in g.adj[u]:
            if g.w(u,v) < 0:
                graph_weights = True
                break
        if graph_weights:
            break
    if not graph_weights:
        return dijkstra_all_pairs(g)
    else:
        return bellman_ford_all_pairs(g)


# In[172]:


g = DWG() 
g.add_node(0)
g.add_node(1)
g.add_node(2)
g.add_node(3) 
"""the graph has directed edges from 0 to 1 and 2, from 1 to 2 and from 2 to 3. 
The other edges will have infinity in their shortest path since the directed edges do not go to them at all."""
g.add_edge(0, 1, 1)
g.add_edge(0, 2, 3)
g.add_edge(1, 2, 1)
g.add_edge(2, 3, 2)
#print(g.adj())
print(g.print_graph())
print("All pairs shortest paths with positive weights:")
print(all_pairs_distance(g))
#print(dijkstra_all_pairs(g))
# g = DWG()
# g.add_node(0)
# g.add_node(1)
# g.add_node(2)
# g.add_node(3)
# g.add_edge(0, 1, 4)
# g.add_edge(0, 2, 3)
# g.add_edge(1, 2, 1)
# g.add_edge(1, 3, 2)
# g.add_edge(2, 3, 5)
# print("All pairs shortest paths with positive weights:")
# print(all_pairs_distance(g))
#Define the graph
# g1 = DWG()
# g1.add_node(0)
# g1.add_node(1)
# g1.add_node(2)
# g1.add_node(3) 
# g1.add_node(4)
# g1.add_edge(0, 1, -1)
# g1.add_edge(0, 2, 4)
# g1.add_edge(1, 2, 3)
# g1.add_edge(1, 3, 2)
# g1.add_edge(1, 4, 2)
# g1.add_edge(3, 2, 5)
# g1.add_edge(3, 1, 1)
# g1.add_edge(4, 3, -3)

# # Test the Bellman-Ford algorithm
# output = all_pairs_distance(g1)
# print(output)
#print(bellman_ford_all_pairs(g1))


# In[ ]:





# In[ ]:





# In[ ]:




