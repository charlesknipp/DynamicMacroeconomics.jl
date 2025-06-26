# DynamicMacroeconomics

A general purpose DSGE interface which presents a lightweight approach to solving macroeconomic models. The purpose of creating this module was to rapidly estimate models in liu of (Childers, 2022) by make DSGE amenable to a probablistic programming paradigm.

## Solving the RBC Model

Consider the basic RBC model

```math
\begin{aligned}
	c_{t+1}^{\gamma} &= c_{t} ^ {\gamma} \beta \left(\alpha z_{t} k_{t}^{\alpha-1} + (1 - \delta) \right) \\
	k_{t} &= z_{t-1} k_{t-1}^{\alpha} - c_{t} + (1 - \delta) k_{t-1} \\
	\log z_{t} &= \nu \log z_{t-1} + \varepsilon
\end{aligned}
```

## First Order Perturbations

Consider a DSGE model with states $y_{t}$ and shock $\varepsilon_{t}$ defined by the optimality conditions $F(y, \varepsilon)$ and it's steady state $y^{*}$ such that $F(y^{*}, 0) = 0$.

Upon taking derivatives, we can achieve a first order approximation of the following form:

```math
0 = A E \left[ y_{t+1} \right] + B y_{t} + C y_{t-1} + D \varepsilon_{t}
```

where A, B, and C are the derivatives with respect to the forward looking, contemporaneous, and lagged state variables respectively and D is the derivative with respect to the shock $\varepsilon_{t}$