import sys
import numpy as np
import random

#Text file containing the graph
GRAPH = sys.argv[1]
#Number of columns in the graph (should be 8 for sample graphs)
COLS = int(sys.argv[2])
#Ending variables (two sets of coordinates)
END1 = [(int(sys.argv[3]), int(sys.argv[4]))]
END2 = [(int(sys.argv[5]), int(sys.argv[6]))]

#amount of pheromone deposited
qADD = 1
#rate of pheromone evaporation
qDECAY = 0.5
#exploration rate
qEXPLORE = 0.9

# i and j are the indices for the node whose neighbors you want to
#Modified from: https://stackoverflow.com/questions/15913489/python-find-all-neighbours-of-a-given-node-in-a-list-of-lists
def find_neighbors(graph, x, y, dist=1):
    neighbors = [row[max(0, x-dist):x+dist+1] for row in graph[max(0, y-1):y+dist+1]]
    neighbors = [elem for nlist in neighbors for elem in nlist]
    directions = [] #direction guide: 0 = left, 1 = up, 2 = right, 3 = down
    #remove diagonal neighbors
    if len(neighbors) == 9:
        for i in [0,1,2,3,4]:
            del neighbors[i]
        directions = [1,0,2,3]
    #if in the top left corner
    elif x == 0 and y == 0:
        for i in [0,1]:
            del neighbors[i]
        directions = [2,3]
    #if in the bottom left corner
    elif x == 0 and y == len(graph) - 1:
        for i in [1,1]:
            del neighbors[i]
        directions = [1,2]
    #if in the top right corner
    elif x == COLS - 1 and y == 0:
        for i in [1,1]:
            del neighbors[i]
        directions = [0,3]
    #if in the bottom right corner
    elif x == COLS-1 and y == COLS-1:
        for i in [0,3]:
            del neighbors[i]
        directions = [1,0]
    #if on the top-most edge
    elif y == 0:
        for i in [1,2,3]:
            del neighbors[i]
        directions = [0,2,3]
    #if on the bottom-most edge
    elif y == len(graph) - 1:
        for i in [0,1,2]:
            del neighbors[i]
        directions = [1,0,2]
    #if on the left-most edge
    elif x == 0:
        for i in [1,1,3]:
            del neighbors[i]
        directions = [1,2,3]
    #if on the right-most edge
    elif x == COLS - 1 :
        for i in [0,2,2]:
            del neighbors[i]
        directions = [1,0,3]
    return neighbors, directions

#Ant Class
class Ant:
    #take in an initial starting position

    def __init__(self, start_x, start_y):
        self.pos = [(start_x,start_y)]
        #last coordinate (to prevent backtracking, unless deadend is 1)
        self.last_coord = []
        #lay pheromone - 1 if laying pheromone, 0 otherwise
        self.pheromone = 1
        #list of nodes visited, in the order visited
        self.visited = []

        if random.randint(0, 1):
            self.end = END1
        else:
            self.end = END2

    #Check the path of the ant
    def path(self):
        return

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

        #sort both lists in increasing weight order
        dirs = [x for _,x in sorted(zip(edges, dirs), reverse=True)]
        edges = sorted(edges, reverse=True)
        #remove all edges that are less than 0
        real_edges = [item for item in edges if item >= 0]
        #necessary for random selection later
        real_edges = np.array(real_edges)
        #add one to all edges (necessary for random selection)
        real_edges += 1
        #remove invalid directions
        diff = len(edges) - len(real_edges)
        if (diff != 0):
            dirs = dirs[0:-diff]

        # print("Current Pos: ", self.pos[0][0], self.pos[0][1])
        # print("Neighbors: ", real_edges)
        # print("Directions: ", dirs)

        #generate random number between 0 and 1
        rn = random.uniform(0,1)
        #choose the next edge
        #base case: only one
        if len(real_edges) == 1:
            self.move(graph, self.pos[0][0], self.pos[0][1], dirs[0], dirs)
        #if all values in the array are the same, randomly choose
        elif (np.unique(real_edges).size == 1):
            self.move(graph, self.pos[0][0], self.pos[0][1], dirs[np.random.choice(np.flatnonzero(a = real_edges.max()))], dirs)
        #otherwise, probabistlically explore
        elif (rn < (1-qEXPLORE)):
            #randomly select max in case of ties
            self.move(graph, self.pos[0][0], self.pos[0][1], dirs[np.random.choice(np.flatnonzero(a = real_edges.max()))], dirs)
        #next step - rn between qEXPLORE and qEXPLORE^2
        elif(rn < qEXPLORE**2):
            #remove maximum value
            real_edges = np.array(list(filter(lambda a: a != max(real_edges), real_edges)))
            dirs = dirs[len(dirs) - len(real_edges):]
            #randomly select max in case of ties
            self.move(graph, self.pos[0][0], self.pos[0][1], dirs[np.random.choice(np.flatnonzero(a = real_edges.max()))], dirs)
        elif(rn < qEXPLORE * (1-qEXPLORE)):
            #remove maximum values twice
            real_edges = np.array(list(filter(lambda a: a != max(real_edges), real_edges)))
            real_edges = np.array(list(filter(lambda a: a != max(real_edges), real_edges)))
            self.move(graph, self.pos[0][0], self.pos[0][1], dirs[np.random.choice(np.flatnonzero(a = real_edges.max()))], dirs)

    #Update pheromone map (if allowed), current ant position, last coordinate, deadend option, visited nodes
    #If the last coord is the only option to move (len(dirs) == 1), deadend
    #If there are more than one option to move (len(dirs) > 2), intersection and pheromone is turned/kept on
    def move(self, graph, x, y, dir, dirs):
        if (len(dirs) >= 2):
            self.pheromone = 1

        #left
        if dir == 0:
            if self.pheromone:
                graph[y][x-1] = graph[y][x-1] + qADD
            if (self.last_coord != [(self.pos[0][0]-1, self.pos[0][1])]):
                self.last_coord = self.pos
                self.visited.append(self.pos[0])
                self.pos = [(self.pos[0][0]-1, self.pos[0][1])]
            else:
                if (len(dirs) == 1):
                    self.pheromone = 0
                self.last_coord = self.pos
        #up
        elif dir == 1:

            if self.pheromone:
                graph[y-1][x] = graph[y-1][x] + qADD
            if (self.last_coord != [(self.pos[0][0], self.pos[0][1]-1)]):
                self.last_coord = self.pos
                self.visited.append(self.pos[0])
                self.pos = [(self.pos[0][0], self.pos[0][1]-1)]
            else:
                if (len(dirs) == 1):
                    self.pheromone = 0
                self.last_coord = self.pos

        #right
        elif dir == 2:
            if self.pheromone:
                graph[y][x+1] = graph[y][x+1] + qADD
            if (self.last_coord != [(self.pos[0][0]+1, self.pos[0][1])]):
                self.last_coord = self.pos
                self.visited.append(self.pos[0])
                self.pos = [(self.pos[0][0]+1, self.pos[0][1])]
            else:
                if (len(dirs) == 1):
                    self.pheromone = 0
                self.last_coord = self.pos

        #down
        elif dir == 3:
            if self.pheromone:
                graph[y+1][x] = graph[y+1][x] + qADD
            if (self.last_coord != [(self.pos[0][0], self.pos[0][1]+1)]):
                self.last_coord = self.pos
                self.visited.append(self.pos[0])
                self.pos = [(self.pos[0][0], self.pos[0][1]+1)]
            else:
                if (len(dirs) == 1):
                    self.pheromone = 0
                self.last_coord = self.pos


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
    for i in range(100):
        numbers = list(range(0,5))
        numbers.append(6)
        ants.append(Ant(random.choice(numbers), END1[0][1]))
    for i in range(50):
        for a in ants:
            #move according to rankedge algorithm
            a.rank_edge(graph)
            #check if the ant has reached the end
            a.check_end()

        #multiply graph by pheromone decay rate
        graph = graph*qDECAY
    print
    for l in graph:
        for n in l:
            print(round(n, 2), end="\t")
        print()



if __name__ == "__main__":
    main()
