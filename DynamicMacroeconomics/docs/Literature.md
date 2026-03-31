# Bibliography

I want to first highlight some essential papers which directly influence the design choices for DynamicMacroeconomics:

- [(Auclert et al, 2021)](https://web.stanford.edu/~aauclert/sequence_space_jacobian.pdf) and [(Auclert et al, 2025)](https://web.stanford.edu/~aauclert/determinacy_sequence_space.pdf) both pioneer macroeconomic modeling in the sequence space via a DAG and define sparse models in terms of Toeplitz operators. Both papers are essential in the various design choices for the modeling syntax as well as the sparse chain rules along the solver.
- [(Rendahl, 2025)](https://onlinelibrary.wiley.com/doi/full/10.1111/ecca.70001) introduces the linear time iteration solver to find the matrix polynomial for one lead and one lag. Furthermore, this algorithm supports automatic differentiation out of the box without the need for custom rules.
- [(Anderson, 1987)](https://www.sciencedirect.com/science/article/pii/0165176585902113) expresses a general form n-leads and m-lags functional form to solve the first order policy function. While their algorithm is largely ignored, the VAR form of the model is used directly in the QZ solver.
- [(Klein, 1999)](https://www.sciencedirect.com/science/article/pii/S0165188999000457) outlines a procedure to remove the forward looking component of a perturbed model using the generalized matrix pencil via QZ decomposition.

Furthermore, there is also literature I seek to reference in the future:

- [(Kao & Hennequin, 2020)](https://arxiv.org/pdf/2011.11430) defines automatic differentiation rules for discrete algebraic Riccati, Lyapunov, and Sylvester equations. This is particularly useful for HMC over a first order perturbed system uses a generalized matrix pencil algorithm to solve the matrix polynomial.
- [(Boehl, 2025)](https://gregorboehl.com/papers/hank_speed_boehl.pdf) uses pushforwards in sequence space to quickly compute nonlinear impulse responses via a Newton method. While I primarily stick to solving the first oder system, the modeling language should be robust enough to support such algorithms.

Lastly, for avoidance of retreading divergent paths, there is also literature which is tangential but I deliberately chose to ignore:

- [(Villamot, 2011)](https://www.dynare.org/wp-repo/dynarewp002.pdf) highlights the process in which Dynare organizes a perturbed system. While this reduces the dimensionality of the problem, it also induces a complex manipulation of the system which departs quite far from the general setting I seek to preserve.
