import sys
import numpy as np
import random
import matplotlib.pyplot as plt

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
qDECAY = .9
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
        for i in [0,2]:
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
        #flag if the ant made it through the path
        self.completed = False
        #number of moves an ant takes
        self.moves = 0
        #path entropy calculation
        self.path_entropy = 1

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
            return True
        if self.pos == self.end and self.end == END2:
            self.end = END1

            return True

    #See if there is a path from one end to the other
    def check_path_end(self):
        self.end=[(END2[0][0]-2, END2[0][1])]
        if self.pos == self.end:
            self.completed = True

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
        #calculate probability of picking the max edge
        if (sum(real_edges) != 0):
            edge_prob = max(real_edges) / sum(real_edges)
        else:
            edge_prob = 0.25
        #multiply current path entropy (if the path is not completed)
        if not self.completed:
            self.path_entropy *= edge_prob
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
        elif(rn < qEXPLORE):
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

    #add pheromone to the coordinate in a specific direction, but do not move there
    def explore(self, graph, x, y, dir):
        if dir == 0:
            if self.pheromone:
                graph[y][x-1] = graph[y][x-1] + qADD
        elif dir == 1:
            if self.pheromone:
                graph[y-1][x] = graph[y-1][x] + qADD
        elif dir == 2:
            if self.pheromone:
                graph[y][x+1] = graph[y][x+1] + qADD
        elif dir == 3:
            if self.pheromone:
                graph[y+1][x] = graph[y+1][x] + qADD

    #Update pheromone map (if allowed), current ant position, last coordinate, deadend option, visited nodes
    #If the last coord is the only option to move (len(dirs) == 1), deadend
    #If there are more than one option to move (len(dirs) > 2), intersection and pheromone is turned/kept on
    def move(self, graph, x, y, dir, dirs):
        #add to the number of moves if it is still searching
        if not self.completed:
            self.moves += 1

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
    #if it is the European Road Network, do this - otherwise...
    tests = ["euro.txt", "air.txt", "usa.txt"]
    if (GRAPH in tests):
        #make a 1174 by 1174 0 matrix
        matrix = np.empty((COLS, COLS))
        #make all coordinates in euro.txt 10s
        with open(GRAPH) as euro:
            coords = []
            for line in euro:
                coords.append([int(i) for i in line.split()])
            for c in coords:
                matrix[c[0]-1, c[1]-1] = 10
        return matrix
    else:
        #X coordinate - always 0
        #Y coordinate - the number of data points along the row
        matrix = np.empty((0,int(COLS)))
        with open(GRAPH) as graph:
            for l in graph:
                #Append the floating point versions of each point in the matrix
                matrix = np.append(matrix, [list(map(float, l.split()))], axis=0)

        return matrix

#Defines if a graph successfully creates a path between both ends
def success_calc(graph):
    total_ants = 100
    #path entropy
    success_ants = 0
    #total moves (will be used for the average)
    total_moves = 0
    #path entropy
    total_path_entropy = 0
    p_e = 1
    ants = []
    move = True
    for i in range(total_ants):
        ants.append(Ant(END1[0][0], END1[0][1]))
    for i in range(500):
        for a in ants:
            a.rank_edge(graph)
            a.check_path_end()

    for a in ants:
        if a.completed:
            success_ants += 1
            total_path_entropy += a.path_entropy
        total_moves += a.moves
    return float(success_ants) / total_ants, float(total_path_entropy) / total_ants, float(total_moves) / total_ants


def main():
    og_graph = read_graph()
    #print(graph)
    plt.imshow(og_graph, cmap="hot", interpolation = "nearest")
    plt.show()
    #repeat analyses
    total_success = 0
    total_path_entropy = 0
    total_moves = 0
    repeat = 10
    for r in range(1):
        print(float(r) / repeat)
        graph = read_graph()
        ants = []
        #Create 100 ants
        for i in range(100):
            numbers = np.where(graph == 10)
            ants.append(Ant(random.choice(numbers[0]), random.choice(numbers[1][:-1])))
        for i in range(500):
            for a in ants:
                #move according to rankedge algorithm
                a.rank_edge(graph)
                #check if the ant has reached the end
                a.check_end()

            #multiply graph by pheromone decay rate
            graph = graph*qDECAY

        success, path_entropy, moves = success_calc(graph)
        total_success += success
        total_path_entropy = path_entropy
        total_moves = moves
        #add graph to og_graph
        og_graph = og_graph + graph

    print("Successful ants: ", total_success / repeat)
    print("Path Entropy: ", total_path_entropy / repeat)
    print("Average moves: ", total_moves / repeat)
    a = np.ma.masked_where(og_graph == 0, og_graph)
    cmap = plt.cm.hot
    cmap.set_bad(color="white")
    plt.imshow(graph,cmap=cmap,interpolation = "nearest")
    plt.show()
    # plt.hist(graph.ravel(), bins=256, range=(0, 200), fc="k", ec="k")
    # plt.show()


if __name__ == "__main__":
    main()
