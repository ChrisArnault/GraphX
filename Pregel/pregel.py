"""pregel.py is a python 2.6 module implementing a toy single-machine
version of Google's Pregel system for large-scale graph processing."""

import collections
import threading
import numpy as np

class Vertex():

    def __init__(self, id, value, out_vertices):
        # This is mostly self-explanatory, but has a few quirks:
        #
        # self.id is included mainly because it's described in the
        # Pregel paper.  It is used briefly in the pagerank example,
        # but not in any essential way, and I was tempted to omit it.
        #
        self.id = id
        self.value = value
        self.out_vertices = out_vertices
        self.incoming_messages = []
        self.outgoing_messages = []

class Pregel():

    def __init__(self, vertices, num_workers):
        self.vertices = vertices
        self.num_workers = num_workers
        self.sum = None
        self.step = 0

    def run(self):
        """Runs the Pregel instance."""
        self.partition = self.partition_vertices()
        while True:
            if self.superstep():
                break
            self.redistribute_messages()

    def partition_vertices(self):
        """Returns a dict with keys 0,...,self.num_workers-1
        representing the worker threads.  The corresponding values are
        lists of vertices assigned to that worker."""
        partition = collections.defaultdict(list)
        for vertex in self.vertices:
            partition[self.worker(vertex)].append(vertex)
        return partition

    def worker(self,vertex):
        """Returns the id of the worker that vertex is assigned to."""
        return hash(vertex) % self.num_workers

    def superstep(self):
        """Completes a single superstep.  

        Note that in this implementation, worker threads are spawned,
        and then destroyed during each superstep.  This creation and
        destruction causes some overhead, and it would be better to
        make the workers persistent, and to use a locking mechanism to
        synchronize.  The Pregel paper suggests that this is how
        Google's Pregel implementation works."""

        values = lambda w : [vertex.value for vertex in w.vertices]
        mat = lambda ws : [np.mat(values(w)).transpose() for w in ws]

        print("============ Start superstep step={}".format(self.step))
        workers = []
        for vertex_list in list(self.partition.values()):
            worker = Worker(vertex_list)
            workers.append(worker)
            worker.start()
        for worker in workers:
            worker.join()
            v = np.sum(values(worker))
            # print("v=", v)

        finished = False

        v = np.sum([np.sum(values(worker)) for worker in workers])
        if self.sum is None:
            pass
        else:
            if abs(v - self.sum) < 0.00001:
                print("========== OK")
                finished = True

        self.sum = v
        self.step += 1

        return finished

    def redistribute_messages(self):
        """Updates the message lists for all vertices."""
        for vertex in self.vertices:
            vertex.incoming_messages = []
        for vertex in self.vertices:
            for (receiving_vertix, message) in vertex.outgoing_messages:
                receiving_vertix.incoming_messages.append((vertex, message))

class Worker(threading.Thread):

    def __init__(self,vertices):
        threading.Thread.__init__(self)
        self.vertices = vertices

    def run(self):
        self.superstep()

    def superstep(self):
        """Completes a single superstep for all the vertices in
        self."""
        for vertex in self.vertices:
            vertex.update()
