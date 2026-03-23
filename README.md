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
<table>
    <tr>
        <td>
            Bioreactor
        </td>
        <td>
            Bioreactor Mayer
        </td>
        <td>
            Bryson Denham
        </td>
        <td>
            Bryson Denham Mayer
        </td>
    </tr>
    <tr>
        <td>
            Cart Pendulum
        </td>
        <td>
            Cart Pendulum Mayer
        </td>
        <td>
            Catalyst Mixing
        </td>
        <td>
            Cushioned Oscillation Mayer
        </td>
    </tr>
    <tr>
        <td>
            Dielectrophoretic Particle Mayer
        </td>
        <td>
            Double Oscillator
        </td>
        <td>
            Double Oscillator Mayer
        </td>
        <td>
            Ducted Fan
        </td>
    </tr>
    <tr>
        <td>
            Egerstedt
        </td>
        <td>
            Egerstedt Mayer
        </td>
        <td>
            Electric Car
        </td>
        <td>
            Electric Car Mayer
        </td>
    </tr>
    <tr>
        <td>
            Fuller
        </td>
        <td>
            Fuller Mayer
        </td>
        <td>
            Hang Glider
        </td>
        <td>
            Hanging Chain
        </td>
    </tr>
    <tr>
        <td>
            Lotka Competitive
        </td>
        <td>
            Lotka Competitive Mayer
        </td>
        <td>
            Lotka Shared
        </td>
        <td>
            Lotka Shared Mayer
        </td>
    </tr>
    <tr>
        <td>
            Lotka Volterra
        </td>
        <td>
            Lotka Volterra Mayer
        </td>
        <td>
            LQR
        </td>
        <td>
            Moon Landing
        </td>
    </tr>
    <tr>
        <td>
            Mountain Car Mayer
        </td>
        <td>
            Ocean
        </td>
        <td>
            Quadrotor
        </td>
        <td>
            Rao Mease
        </td>
    </tr>
    <tr>
        <td>
            Three Tank
        </td>
        <td>
            Three Tank Mayer
        </td>
        <td>
            Van der Pol
        </td>
        <td>
            Van der Pol Mayer
        </td>
    </tr>
    <tr>
        <td>
            Lotka OED
        </td>
        <td>
            Dielectr Particle OED
        </td>
        <td>
            Jackson OED
        </td>
        <td>
            Van der Pol OED
        </td>
    </tr>
</table


## Using the new BlockSQP
To use the new version of BlockSQP instead of the one included in CasADi, you have to change the path inside the file `blocksqp_path.txt`:
```
# path to your local installation of BlockSQP 2:
{MY_PATH}/blocksqp/python_Interface
```

## Using the CasADi BlockSQP
To use the CasADi version of BlockSQP, you have to configure the [ma27 solver](https://www.hsl.rl.ac.uk/ipopt/).

