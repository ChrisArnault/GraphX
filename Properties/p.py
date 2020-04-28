
import random
import numpy as np

N = 10
P = 10

g = np.array([3]) * np.array([np.array([(o, p) for p in random.sample(range(P), random.randint(0, P))])[:, np.newaxis] for o in range(N)])
