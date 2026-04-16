# Tuning Algorithms — Mathematical Background

## Grid Search

Exhaustive evaluation of all combinations in a discrete HP grid. If $k$ parameters each have $n_i$ values:

$$\text{Total trials} = \prod_{i=1}^{k} n_i$$

Guarantees finding the optimal combination within the grid but scales exponentially.

## Random Search

Sample HP values independently from specified distributions. For a budget of $N$ trials, each trial draws:

$$\theta^{(t)} \sim \prod_{i=1}^{k} p_i(\theta_i)$$

where $p_i$ can be uniform, log-uniform, or categorical. More efficient than grid search when only a few HPs matter (Bergstra & Bengio, 2012).

## Bayesian Optimization (TPE)

Tree-structured Parzen Estimator models the objective function using two density estimators:

$$l(\theta) = p(\theta | y < y^*)$$
$$g(\theta) = p(\theta | y \geq y^*)$$

The next trial maximizes the Expected Improvement ratio $l(\theta)/g(\theta)$.

Reference: Bergstra et al. (2011)

## Genetic Algorithm

Population-based search with evolutionary operators:

**Tournament selection:** Select parents by choosing the best from a random subset of size $k$.

**BLX-alpha crossover** (for continuous parameters):

$$c = \text{Uniform}(\min(p_1,p_2) - \alpha \cdot d, \max(p_1,p_2) + \alpha \cdot d)$$

where $d = |p_1 - p_2|$ and $\alpha$ is typically 0.5.

**Gaussian mutation** (for continuous parameters):

$$\theta' = \theta + \mathcal{N}(0, \sigma^2)$$

where $\sigma$ is proportional to the parameter range.

**Elitism:** Preserve the top $k$ individuals unchanged across generations.

## Temporal Cross-Validation

All validation splitters respect temporal ordering:

**LastBlock:** Train on $[1, T-v]$, validate on $[T-v+1, T]$

**ExpandingValidation:** For $t = t_{min}, ..., T-1$: train on $[1, t]$, validate on $[t+g+1]$ where $g$ is the embargo gap.

**BlockedKFold:** Divide $[1, T]$ into $K$ blocks. For each fold, train on all blocks except one, validate on the held-out block, with embargo gaps.

**See also:** [User Guide: Tuning](../user_guide/tuning.md) — when to use each algorithm
