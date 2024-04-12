#!/usr/bin/env python
# coding: utf-8

# # Part 3
# In this part, you will analyze and experiment with a modification of Dijkstra’s algorithm called the A*
# (we will cover this algorithm in next lecture, but you are free to do your own research if you want to get
# started on it). The algorithm essentially, is an “informed” search algorithm or “best-first search”, and is
# helpful to find best path between two given nodes. Best path can be defined by shortest path, best time, or
# least cost. The most important feature of A* is a heuristic function that can control it’s behavior.
# 
# 
# Part 3.1: Write a function A_Star (graph, source, destination, heuristic) which takes in a directed weighted
# graph, a sources node, a destination node , and a heuristic “function”. Assume h is a dictionary which
# takes in a node (an integer), and returns a float. Your method should return a 2-tuple where the first
# element is a predecessor dictionary, and the second element is the shortest path the algorithm determines
# from source to destination. This implementation should be using priority queue.

# In[1]:


#weighted digraph
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


# In[2]:


class Item:
    def __init__(self, value, key):
        self.key = key
        self.value = value
    
    def __str__(self):
        return "(" + str(self.key) + "," + str(self.value) + ")"
class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.build_heap()

        # add a map based on input node
        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self,index):
        return 2 * (index + 1) - 1

    def find_right_index(self,index):
        return 2 * (index + 1)

    def find_parent_index(self,index):
        return (index + 1) // 2 - 1  
    
    def sink_down(self, index):
        smallest_known_index = index

        if self.find_left_index(index) < self.length and self.items[self.find_left_index(index)].key < self.items[index].key:
            smallest_known_index = self.find_left_index(index)

        if self.find_right_index(index) < self.length and self.items[self.find_right_index(index)].key < self.items[smallest_known_index].key:
            smallest_known_index = self.find_right_index(index)

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            
            # update map
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index

            # recursive call
            self.sink_down(smallest_known_index)

    def build_heap(self,):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink_down(i) 

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def insert_nodes(self, node_list):
        for node in node_list:
            self.insert(node)

    def swim_up(self, index):
        
        while index > 0 and self.items[self.find_parent_index(index)].key < self.items[self.find_parent_index(index)].key:
            #swap values
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            #update map
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def get_min(self):
        if len(self.items) > 0:
            return self.items[0]

    def extract_min(self,):
        #xchange
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        #update map
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.map.pop(min_node.value)
        self.sink_down(0)
        return min_node

    def decrease_key(self, value, new_key):
        if new_key >= self.items[self.map[value]].key:
            return
        index = self.map[value]
        self.items[index].key = new_key
        self.swim_up(index)

    def get_element_from_value(self, value):
        return self.items[self.map[value]]
    
    def __contains__(self, value):
        for item in self.items:
            if item.value == value:
                return True
        return False

    def is_empty(self):
        return self.length == 0
    
    def __str__(self):
        height = math.ceil(math.log(self.length + 1, 2))
        whitespace = 2 ** height + height
        s = ""
        for i in range(height):
            for j in range(2 ** i - 1, min(2 ** (i + 1) - 1, self.length)):
                s += " " * whitespace
                s += str(self.items[j]) + " "
            s += "\n"
            whitespace = whitespace // 2
        return s


# In[3]:


import math
def A_star(graph, source, destination, h):
    
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
                if neighbour not in openset:
                    openset.insert(Item(neighbour, fscore[neighbour]))
    shortestpath = []
    current = destination
    while current in cameFrom:
        shortestpath.insert(0, current)
        current = cameFrom[current]
    shortestpath.insert(0, source)
    return cameFrom, shortestpath
    


# In[4]:


graph3 = DWG()
graph3.add_node(0)
graph3.add_node(1)
graph3.add_node(2)
graph3.add_node(3)
graph3.add_node(4)
graph3.add_node(5)
graph3.add_edge(0,1,1)
graph3.add_edge(0,2,4)
graph3.add_edge(1,2,2)
graph3.add_edge(1,3,5)
graph3.add_edge(2,3,1)
source = 0
destination = 2
h = {node : float(graph3.heuristic(node, destination)) for node in graph3.adj.keys()}
A_star(graph3, 0, 2, h)


# In[5]:


graph2 = DWG()
graph2.add_node(0)
graph2.add_node(1)
graph2.add_node(2)
graph2.add_node(3)
graph2.add_node(4)
graph2.add_node(5)
graph2.add_edge(0,1,1)
graph2.add_edge(0,2,4)
graph2.add_edge(1,2,2)
graph2.add_edge(1,3,5)
graph2.add_edge(2,3,1)
graph2.add_edge(3,5,1)
graph2.add_edge(0,5,2)
source = 0
destination = 5
h = {node : float(graph2.heuristic(node, destination)) for node in graph2.adj.keys()}
A_star(graph2, source, destination, h)

