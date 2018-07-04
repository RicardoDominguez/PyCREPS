import numpy as np

class Plant:
    def rollout(self, hpol, pol):
        x = 0
        H = 10
        w = hpol.mu
        latent = []
        for t in xrange(H):
            u = pol.sample(w, x)
            if isinstance(u, np.ndarray):
                u = u[0]
            latent.append([x, u])
            y = x + u
            x = y
        latent.append([y, u])
        latent = np.array(latent)
        X = latent[0:-1, :]
        Y = latent[1:, :]
        return X, Y
