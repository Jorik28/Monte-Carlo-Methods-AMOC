# Sampling a bistable ocean-circulation model, using three Monte Carlo Methods: RW-MH, MALA and HMC

Course project for Laboratory Class in Computational Methods, Utrecht University, 2026.

This project implements and compares three Markov chain Monte Carlo samplers
on a simplified model of the Atlantic Meridional Overturning Circulation
(AMOC), which can run in two stable modes. The samplers are first validated on
known distributions, then used on the bistable target, and finally used for
Bayesian inference of the model parameters from noisy observations. Convergence
and efficiency are assessed with R-hat and the effective sample size (ESS).

## The model

The equilibrium condition is $\lambda = |a-y|\cdot y$, where $y$ is the rescaled
salinity difference driving the circulation, $\lambda$ the forcing, and $a$ a
temperature-difference threshold. The dynamics is a gradient flow with
potential $V(y)$. With noise of strength $\sigma$, the stationary distribution
is $\pi(y) \propto \exp(-2V(y)/\sigma^2)$, which has two modes for suitable
$(\lambda, a)$.

The project uses this model in two ways:

1. **Sampling $y$** for fixed $(\lambda, a)$ (target: $\pi(y)$, 1D).
2. **Bayesian inference** of $(\lambda, a)$ from a few noisy observations of
   the equilibrium $y_{eq}(\lambda, a)$ (target: the posterior, 2D).

## Repository structure

```
amoc_model.py    Potential V(y), equilibrium condition g(y), Newton solver y_eq
samplers.py      Sampler library: random_walk_metropolis, mala, hmc, r_hat, 
		 effective_sample_size
project_1.ipynb  Full analysis: validation, experiments, discussion
```

The sampler functions are generic: they only need `log_prob` (and
`grad_log_prob`) and know nothing about the AMOC model. Model-specific code
lives in `amoc_model.py`; the notebook connects the two.

## What is implemented

- **Newton solver** for the equilibria of $g(y)=0$, with a starting guess that
  keeps it on the upper stable branch, checked against the closed-form root.
- **Random-walk Metropolis-Hastings (RW-MH)**, validated against a standard
  normal and against a grid-normalised $\pi(y)$.
- **Metropolis-Adjusted Langevin Algorithm (MALA)**, with the proposal-density 
  correction in the acceptance step.
- **Hamiltonian Monte Carlo (HMC)** with leapfrog integration; gradients of the 
  log-posterior via the
  chain rule and central finite differences for $\partial y_{eq}/\partial\theta$.
- **Diagnostics**: Gelman-Rubin R-hat and ESS (autocorrelation-based), both
  validated on cases with known answers, applied to at least 4 chains per method
  started from spread-out points.

## Main findings

- **Validation:** all samplers reproduce the standard normal and the grid
  density of $\pi(y)$.
- **Convergence:** R-hat is $<1.03$ for RW-MH and MALA. With the initial
  $(\varepsilon, L) = (0.01,15)$, two HMC chains failed because the gradient 
  was too steep at the starting points; reducing $\varepsilon$ to $0.0015$ 
  (with $L$ increased to keep $\varepsilon \cdot L$ fixed) fixed this.
- **Efficiency:** HMC reaches an ESS per second of roughly 1,5 times that of RW-MH 
  and 4.5 times that of MALA.

## Running the code

Requirements: Python 3, `numpy`, `matplotlib`, and Jupyter.

```
pip install numpy matplotlib jupyter
jupyter notebook project_1.ipynb
```

## Notes

- Costs are compared in seconds, this depends on the specific computational setup.
- Results depend on the seed and on the tuning of $\varepsilon$ and $L$ for HMC.