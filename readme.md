# Contextual Relative Entropy Policy Search

Python implementation of Contextual Relative Entropy Policy Search for Reinforcement Learning problems.

Relevant papers:
 * Kupcsik, A. G., Deisenroth, M. P., Peters, J., & Neumann, G. (2013, July). Data-efficient generalization of robot skills with contextual policy search. In Proceedings of the 27th AAAI Conference on Artificial Intelligence, AAAI 2013 (pp. 1401-1407). [[pdf]](https://www.aaai.org/ocs/index.php/AAAI/AAAI13/paper/viewFile/6322/6842)
 * Peters, J., Mülling, K., & Altun, Y. (2010, July). Relative Entropy Policy Search. In AAAI (pp. 1607-1612).[[pdf]](http://www.aaai.org/ocs/index.php/AAAI/AAAI10/paper/viewFile/1851/2264)
 * Daniel, C., Neumann, G., & Peters, J. (2012, March). Hierarchical relative entropy policy search. In Artificial Intelligence and Statistics (pp. 273-281). [[pdf]](http://www.jmlr.org/proceedings/papers/v22/daniel12/daniel12.pdf)
 * Daniel, C., Neumann, G., Kroemer, O., & Peters, J. (2016). Hierarchical relative entropy policy search. The Journal of Machine Learning Research, 17(1), 3190-3239. [[pdf]](http://www.jmlr.org/papers/volume17/15-188/15-188.pdf)
 * Kupcsik, A., Deisenroth, M. P., Peters, J., Loh, A. P., Vadakkepat, P., & Neumann, G. (2017). Model-based contextual policy search for data-efficient generalization of robot skills. Artificial Intelligence, 247, 415-439. [[pdf]](http://eprints.lincoln.ac.uk/25774/1/Kupcsik_AIJ_2015.pdf)

## Implementation

Contextual Policy Search allows to generalize policies to multiple contexts, where the context contains some information about the Reinforcement Learning problem at hand, such as the objective of the agent or properties of the environment. CREPS follows a hierarchical approach to contextual policy search, in which there are two policies:
 * A lower policy __&pi;(u | x; w)__ which determines the action __u__ taken by the agent given its state __x__ and some parameters __w__.
 * An upper policy __&pi;(w | s)__  which determines the lower policy parameters __w__ given the context __s__.

The file *CREPS.py* implements the upper policy as a linear-Gaussian model which is updated using weighted ML according to the contexts and rewards of the observed episodes.

All other elements of the Reinforcement Learning problem &mdash;environment dynamics, reward function and lower policy&mdash; must be implemented for your particular scenario as you consider best with only a few considerations in mind to ensure compatibility with the upper policy and policy update function. It is then very straightforward to put everything together, as illustrated in the next section.

## How to set up your own scenario

## Examples

## Contributing...

All enhancements are welcome. Feel free to give suggestions or raise issues.
