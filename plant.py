import numpy as np
from simulator import Scenario

class Plant:
    def rollout(self, x0, H, hpol, pol, cost):
        w = hpol.mu.reshape(-1, 1)
        R = 0

        scn = Scenario(0.1)
        scn.initScenario(x0)
        x = x0
        for t in xrange(H): # For each step within horizon
            u = pol.sample(w, x)
            y = scn.step(u)
            R += cost.sample(y)
            x = y
            #scn.plot()

        return R[0, 0]
