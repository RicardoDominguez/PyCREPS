import numpy as np
import pdb

class Plant:
    def rollout(self, scn, x0, H, hpol, pol, cost):
        w = hpol.contextMean(x0.reshape(1, -1))
        # pdb.set_trace()
        pol.reset()
        R = 0

        scn.initScenario(x0)
        x = x0
        for t in xrange(H): # For each step within horizon
            u = pol.sample(w, x)
            y, ry = scn.step(u)
            #pdb.set_trace()
            R += cost.sample(ry)
            x = y
            #scn.plot()

        return R[0, 0]
