# Contextual Relative Entropy Policy Search

Python implementation of Contextual Relative Entropy Policy Search for Reinforcement Learning problems.

- [Relevant papers](#relevant-papers)
- [Implementation](#implementation)
- [How to set up your own scenario](#how-to-set-up-your-own-scenario)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## Relevant papers
 * Kupcsik, A. G., Deisenroth, M. P., Peters, J., & Neumann, G. (2013, July). Data-efficient generalization of robot skills with contextual policy search. In Proceedings of the 27th AAAI Conference on Artificial Intelligence, AAAI 2013 (pp. 1401-1407). [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/viewFile/6322/6842)
 * Peters, J., Mülling, K., & Altun, Y. (2010, July). Relative Entropy Policy Search. In AAAI (pp. 1607-1612).[[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1851/2264)
 * Daniel, C., Neumann, G., & Peters, J. (2012, March). Hierarchical relative entropy policy search. In Artificial Intelligence and Statistics (pp. 273-281). [[pdf]](http://www.jmlr.org/proceedings/papers/v22/daniel12/daniel12.pdf)
 * Daniel, C., Neumann, G., Kroemer, O., & Peters, J. (2016). Hierarchical relative entropy policy search. The Journal of Machine Learning Research, 17(1), 3190-3239. [[pdf]](http://www.jmlr.org/papers/volume17/15-188/15-188.pdf)
 * Kupcsik, A., Deisenroth, M. P., Peters, J., Loh, A. P., Vadakkepat, P., & Neumann, G. (2017). Model-based contextual policy search for data-efficient generalization of robot skills. Artificial Intelligence, 247, 415-439. [[pdf]](http://eprints.lincoln.ac.uk/25774/1/Kupcsik_AIJ_2015.pdf)

## Implementation

Contextual Policy Search allows to generalize policies to multiple contexts, where the context contains some information about the Reinforcement Learning problem at hand, such as the objective of the agent or properties of the environment. CREPS follows a hierarchical approach to contextual policy search, in which there are two policies:
 * A lower policy __&pi;(u | x; w)__ which determines the action __u__ taken by the agent given its state __x__ and some parameters __w__.
 * An upper policy __&pi;(w | s)__  which determines the lower policy parameters __w__ given the context __s__.

The file [CREPS.py](CREPS.py) implements the upper policy as a linear-Gaussian model which is updated using weighted ML.

All other elements of the Reinforcement Learning problem &mdash;environment dynamics, reward function and lower policy&mdash; must be implemented for your particular scenario as you consider best with only a few considerations in mind to ensure compatibility with the upper policy and policy update function. It is then very straightforward to put everything together, as illustrated in the next section.

Implementations of [CREPS.py](CREPS.py) using PyTorch ([CREPS_torch.py](CREPS_torch.py)) and Theano ([CREPS_theano.py](CREPS_theano.py)) are provided, but please ensure that your application is sufficiently computationally expensive to take advantage of these methods (otherwise the computational overhead introduced will be larger than the run-time performance improvements).

## How to set up your own scenario

The steps of the each policy iteration are:

```
1. Run M episodes, storing for each episode the lower policy parameters sampled,
   the episode context and the episodic reward
2. Compute the sample weights for policy update
3. Update upper policy using the sample weights
```

Which in my examples I have implemented as:

```
R, W, F = predictReward(env, M, hpol, pol) # 1
p = computeSampleWeighting(R, F, eps)      # 2
hpol.update(W, F, p)                       # 3
```
where ``env`` is a class with the environment dynamics, ``hpol`` the upper policy and ``pol`` the lower level policy.

The methods for steps 2 and 3 are implemented in [*CREPS.py*](CREPS.py), thus you only need to worry about step 1, which __MUST__ return the following numpy arrays:
 * ``R`` - reward for each episode, *shape (M,)*
 * ``W`` - lower policy weights used for each episode, *shape (M x number of lower policy weights)*
 * ``F`` - context of each episode. *shape (M x number of context parameters)*

Imagine your scenario is an OpenAI Gym environment, an intuitive implementation of ``predictReward`` would be:

```
def predictReward(env, M, hipol, pol):
  for rollout in xrange(M):
      s = env.reset()                    # Sample context
      w = hipol.sample(s.reshape(1, -1)) # Sample lower-policy weights

      W[rollout, :] = w
      F[rollout, :] = s

      done = False
      x = s
      while not done:
          u = pol.sample(w.T, x)          # Sample action from lower policy
          x, r, done, info = env.step(u)
          R[rollout] += r                 # Update episode reward

  return R, W, F
  ```

## Examples

In all the examples provided there are three files:
 * Script called to train the policy for the specific scenario (foo_learn.py)
 * File containing functions/classes which implement the environment dynamics, reward function and lower policy (scenario.py)
 * File with some functions to benchmark the performance of the algorithm at each policy update (benchmarks.py)

For a full example of CREPS being used to solve the [Cart Pole](https://gym.openai.com/envs/CartPole-v0/) OpenAI gym environment check [/cartPole](/cartPole). This example includes the optional use of PyTorch and Theano through the global flags ``use_torch`` and ``use_theano``. To run it use:
```
$ python cartPole/cartPole_learn.py
```

For a full example of CREPS being used to solve the [Acrobot](https://gym.openai.com/envs/Acrobot-v1/) OpenAI gym environment check [/acrobot](/acrobot). To run it use:
```
$ python acrobot/acrobot_learn.py
```

For a full example of how you could use CREPS for your own Reinforcement Learning problem check [/customEnv](/customEnv), where a differential drive robot learns to follow a straight wall using a PID controller (here the context is the starting distance from the wall and initial angle with respect to the wall). To run it use
```
$ python customEnv/robot_learn.py
```

Furthermore, CREPS can be easily extended to a more data-efficient model-based approach. [/cartPole_GPREPS](/cartPole_GPREPS) offers a quick example of this approach, using Gaussian Processes to learn the forward dynamics of the environment. To run it use
```
$ python cartPole_GPREPS/cartPole_learn.py
```


## Dependencies
 * ``numpy``
 * ``scipy``


Optional:
 * ``pytorch``
 * ``theano``

 For the examples:
 * ```gym```
 * ```matplotlib```

## Contributing

All enhancements are welcome. Feel free to give suggestions or raise issues.
