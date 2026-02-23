# Python software for Strategies and Heuristics Regarding Lifting of Optimal Control problems (PySHeRLOC)

## Getting started
First, create a new virtual python environment, e.g.,
```
python -m venv .venv
```
Then activate the virtual environment and install the necessary packages via:
```
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Now you can start the solver with user interface by running 
```
python GUI.py
```
from the same folder as the file itself.

![User Interface on MacOS](assets/gui.png)

The Problems can be solved using IPOPT, blockSQP, blockSQP2, or fatrop.
Benchmark problems can be found inside the `Apps` folder. Those include, among many others:
- Batch Reactor
- Bioreactor
- Catalyst Mixing
- D'Onofrio
- Egerstedt
- Fuller 
- Hanging Chain
- Lotka Volterra
- Lotka Volterra OED
- Quadrotor
- Stirred Tank Reactor
- Van der Pol


## Using the new BlockSQP
To use the new version of BlockSQP instead of the one included in CasADi, you have to change the path inside the file `blocksqp_path.txt`:
```
# path to your local installation of BlockSQP 2:
{MY_PATH}/blocksqp/python_Interface
```

## Using the CasADi BlockSQP
To use the CasADi version of BlockSQP, you have to configure the [ma27 solver](https://www.hsl.rl.ac.uk/ipopt/).

