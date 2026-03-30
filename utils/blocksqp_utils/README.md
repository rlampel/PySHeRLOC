# Brief file descriptions

## auto_condensing_heuristic.py
Implements Algorithm 2 from [1].

## blocksqp_init_heuristics.py
Implements Algorithm 1 from [1].

## blocksqp_options.py
Collection of options for blockSQP2 [2].

## create_blocksqp_problem.py
Contains the main function that solves the problem using blockSQP2 and implements callbacks for the algorithms from [1].

## create_condenser.py
Function to condense the lifted problem, see [2].

## dyn_lifting.py
Implements the merit-based initialization via dynamic programming from [1].

## fast_init_lift.py
Implements the more efficient version of the function from `dyn_lifting.py`, assuming that all controls are constant.

## fsinit_eval.ps
Computes the states, constraint violations, and objective for FSInit.

## get_block_sizes.py
Computes block sizes that are needed for condensing.

## get_blocksqp_path.py
Sets the path to the local installation of blockSQP as defined in `blocksqp_path.txt`.

## log_conv_data.py
Functions to save and print various metrics for convergence.

## sort_vars.py
Fuctions to sort the primal variables to obtain a block structure.

## References
[1]: [Lampel, R., Sager, S.: "On lifting strategies for optimal control problems"](https://optimization-online.org/?p=34160)

[2]: [Wittmann, R.: "blockSQP2"](https://github.com/rlampel/blockSQP2)
