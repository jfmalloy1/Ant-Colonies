import sys
import numpy as np
import random

#Text file containing the graph
GRAPH = sys.argv[1]
#Number of columns in the graph (should be 8 for sample graphs)
COLS = sys.argv[2]
#Ending variables (two sets of coordinates)
END1 = [(int(sys.argv[3]), int(sys.argv[4]))]
END2 = [(int(sys.argv[5]), int(sys.argv[6]))]

#amount of pheromone deposited
qADD = 1
#rate of pheromone evaporation
qDECAY = 0.1
#exploration rate
qEXPLORE = 0.1

# i and j are the indices for the node whose neighbors you want to
#Modified from: https://stackoverflow.com/questions/15913489/python-find-all-neighbours-of-a-given-node-in-a-list-of-lists
#FIX THIS
def find_neighbors(graph, x, y, dist=1):
    neighbors = [row[max(0, x-dist):x+dist+1] for row in graph[max(0, y-1):y+dist+1]]
    neighbors = [elem for nlist in neighbors for elem in nlist]
    directions = [] #direction guide: 0 = left, 1 = up, 2 = right, 3 = down
    #remove diagonal neighbors
    if len(neighbors) == 9:
        for i in [0,1,2,3,4]:
            del neighbors[i]
        directions = [0,1,2,3]
    elif x == 0:
        for i in [1,1,3]:
            del neighbors[i]
        directions = [1,2,3]
    elif x == 7:
        for i in [0,2,2]:
            del neighbors[i]
        directions = [0,1,3]
    return neighbors, directions

#Ant Class
class Ant:
    #take in an initial starting position

    def __init__(self, start_x, start_y):
        self.pos = [(start_x,start_y)]
        #last coordinate (to prevent backtracking, unless deadend is 1)
        self.last_coord = []
        #deadend - 0 if not coming from a deadend, 1 if so
        self.deadend = 0
        #intersection - 1 if at an intersection, 0 if not
        self.intersection = 0
        #lay pheromone - 1 if laying pheromone, 0 otherwise
        self.lay_pheromone = 1
        #list of nodes visited, in the order visited
        self.visited = [(start_x, start_y)]

        if random.randint(0, 1):
            self.end = END1
        else:
            self.end = END2

    #Check the path of the ant
    def path(self):
        print(self.visited)

    #If the ant reaches the end, turn around & search for the other end
    def check_end(self):
        if self.pos == self.end and self.end == END1:
            self.end = END2
        if self.pos == self.end and self.end == END2:
            self.end = END1

    #rank edge algorithm
    def rank_edge(self, graph):
        #potential edges to move to (sorted by edge weights)
        edges, dirs = find_neighbors(graph, self.pos[0][0], self.pos[0][1])
        print("Neighbors: ", edges)
        print("Directions: ", dirs)

        #sort both lists
        dirs = [x for _,x in sorted(zip(edges, dirs), reverse=True)]
        edges = sorted(edges, reverse=True)
        print(edges, dirs)


#Read the graph from a text file given as input
def read_graph():
    #X coordinate - always 0
    #Y coordinate - the number of data points along the row
    matrix = np.empty((0,int(COLS)))
    with open(GRAPH) as graph:
        for l in graph:
            #Append the floating point versions of each point in the matrix
            matrix = np.append(matrix, [list(map(float, l.split()))], axis=0)

    return matrix

def main():
    graph = read_graph()
    print(graph)
    #Create 100 ants
    ants = []
    for i in range(1):
        ants.append(Ant(random.randint(0, int(COLS)-1), 2))
    for a in ants:
        a.path()
        a.rank_edge(graph)



if __name__ == "__main__":
    main()
