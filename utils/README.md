# Brief file descriptions

## adapt_init.py
Functions to compute the Lagrange multipliers that minimize the KKT error for given primal variables and the corresponding active set.

## create_nlp.py
Function which takes an instance of a problem class and given discretization grids as inputs. It returns state variables, controls, bounds, as well as the objective function that are needed as inputs for the solver.

## get_problem.py
Functions that return a problem instance that matches the given input name.

## initialization.py
Methods to initialize the state variables (constant / automatic / random).

## ode_solver.py
Collection of methods to solve ordinary differential equations.

## penalty.py
Functions to compute constraint violations and penalties.

## plot_solution.py
Function for plotting states and controls.

## sensitivity.py (not recommended)
Implements different approaches to introduce lifting points based on the computed sensitivities.

## References
[1]: [Lampel, R., Sager, S.: "On lifting strategies for optimal control problems"](https://optimization-online.org/?p=34160)

