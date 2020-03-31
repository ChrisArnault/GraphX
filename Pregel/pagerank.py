"""pagerank.py illustrates how to use the pregel.py library, and tests
that the library works.

It illustrates pregel.py by computing the PageRank for a randomly
chosen 10-vertex web graph.

It tests pregel.py by computing the PageRank for the same graph in a
different, more conventional way, and showing that the two outputs are
near-identical."""

from pregel import Vertex, Pregel

# The next two imports are only needed for the test.
import numpy as np
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d


num_workers = 4
num_vertices = 1000
distance_max = 0.1


def main():
    vertices = [PageRankVertex(j, 1.0/num_vertices, [])
                for j in range(num_vertices)]
    X = [vertices[j].x for j in range(num_vertices)]
    Y = [vertices[j].y for j in range(num_vertices)]

    create_edges(vertices)

    pr_test = pagerank_test(vertices)
    # print("Test computation of pagerank:\n%s" % pr_test)

    p = Pregel(vertices, num_workers)


    pr_pregel = pagerank_pregel(p)
    # print("Pregel computation of pagerank:\n%s" % pr_pregel)

    diff = pr_pregel-pr_test
    # print("Difference between the two pagerank vectors:\n%s" % diff)
    print("The norm of the difference is: %s" % np.linalg.norm(diff))

    plt.show()

def create_edges(vertices):
    """Generates 4 randomly chosen outgoing edges from each vertex in
    vertices."""

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'lightgray', 'rosybrown', 'orange', 'lightgreen', 'lightblue', 'lightpink']
    cindex = 0
    for vertex in vertices:

        """
        Building vertices connected to a given vertex:
        - select a random number of possible vertices from the whole set
        - from this selection, select only vertices closer to it than distance_max
        """
        # vertex.out_vertices = random.sample(vertices, np.random.randint(0, len(vertices)))

        vertex.out_vertices = [v for v in random.sample(vertices, np.random.randint(0, np.sqrt(len(vertices)))) if vertex.dist(v) < distance_max]

        out = ",".join(["{}".format(v.id) for v in vertex.out_vertices])

        # print("create_edges> id={} out={}".format(vertex.id, out))

        plt.scatter(vertex.x, vertex.y, c=colors[cindex], s=1)

        for v in vertex.out_vertices:
            plt.plot([vertex.x, v.x], [vertex.y, v.y], colors[cindex])

        cindex += 1
        if cindex >= len(colors):
            cindex = 0

def pagerank_test(vertices):
    """Computes the pagerank vector associated to vertices, using a
    standard matrix-theoretic approach to computing pagerank.  This is
    used as a basis for comparison."""
    I = np.mat(np.eye(num_vertices))
    G = np.zeros((num_vertices,num_vertices))
    for vertex in vertices:
        num_out_vertices = len(vertex.out_vertices)
        for out_vertex in vertex.out_vertices:
            G[out_vertex.id,vertex.id] = 1.0/num_out_vertices
    P = (1.0/num_vertices)*np.mat(np.ones((num_vertices,1)))
    return 0.15*((I-0.85*G).I)*P

def pagerank_pregel(p):
    """Computes the pagerank vector associated to vertices, using
    Pregel."""
    p.run()
    return np.mat([vertex.value for vertex in p.vertices]).transpose()

class PageRankVertex(Vertex):

    def __init__(self, id, value, out_vertices):
        Vertex.__init__(self, id, value, out_vertices)
        self.x = np.random.random()
        self.y = np.random.random()

    def dist(self, other) -> float:
        return np.sqrt(pow(self.x - other.x, 2.0) + pow(self.y - other.y, 2.0))

    def update(self):
        # This routine has a bug when there are pages with no outgoing
        # links (never the case for our tests).  This problem can be
        # solved by introducing Aggregators into the Pregel framework,
        # but as an initial demonstration this works fine.

        self.value = 0.15 / num_vertices + 0.85 * sum(
            [pagerank for (vertex, pagerank) in self.incoming_messages])
        incoming = "-".join(["({},{})".format(vertex.id, pagerank) for (vertex, pagerank) in self.incoming_messages])
        outgoing = ""
        try:
            outgoing_pagerank = self.value / len(self.out_vertices)
            self.outgoing_messages = [(vertex, outgoing_pagerank) for vertex in self.out_vertices]
            outgoing = "-".join(["({},{})".format(vertex.id, outgoing_pagerank) for vertex in self.out_vertices])
        except:
            pass
            # print("update> superstep={} id={} in=[{}] out=[{}]".format(self.superstep, self.id, incoming, outgoing))

if __name__ == "__main__":
    main()
