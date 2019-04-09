import sys
import numpy as np

#Text file containing the graph
GRAPH = sys.argv[1]
#Number of columns in the graph
COLS = sys.argv[2]

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


if __name__ == "__main__":
    main()
