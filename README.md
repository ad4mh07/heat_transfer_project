# heat_transfer_project

A Python-based  simulation of heat dispersion inside a uniform cube, using finite difference methods and Laplace's heat equation to study how boundary conditions influence the temperature field throughout cube, and providing an interactive visualisation of the result.

## Overview

This project models steady-heat conduction in a uniform cube by numerically solving Laplace's equation in 3 dimensions. This is done by modelling each of the interior nodes as an average of its neighbours, to iteratively converge to the solution, subject to the user's defined boundary conditions

The simulation includes:
- Variable cube resolution
- User-defined boundary conditions
- Three dimensional coloured temperature visualisation
- Zoomed-in views of interior nodes
- Two dimensional cross-sectional slicing in any axis 

This project was developed to explore the finite difference method, and using numerical methods for solving PDEs, alongside how resolution impacts accuracy and computational performance

The temperature field satisfies Laplace’s equation,

$$
\nabla^2 T = 0
$$

which, in Cartesian coordinates, becomes

$$
\frac{\partial^2 T}{\partial x^2}
+
\frac{\partial^2 T}{\partial y^2}
+
\frac{\partial^2 T}{\partial z^2}
=0.
$$

Using a finite difference approximation, each interior grid point is updated according to the average of its six neighbouring points, 

$$
T_{i,j,k} = \frac{
T_{i+1,j,k}
+
T_{i-1,j,k}
+
T_{i,j+1,k}
+
T_{i,j-1,k}
+
T_{i,j,k+1}
+
T_{i,j,k-1}
}{6}
$$

The solution is obtained iteratively until convergence.

## Repository structure 
  README- the README for this project
  simulation- the final, tidied up iteration of this project
  example- an example of 6 well-defined input functions, and 3 calls of the simulation
  figures- saved figures of the example
  models- a rough python script containing all major iterations of the project. The final iteration, 4, is nearly identical to the script in 'simulation.py', just a bit less polished
  timing- a script containing a modified version of the simulation that measures the computational performance (speed) at different resolutions
  timing_report- a short tables of results I achieved from the timing.py script

## Features

### Numerical Solution
The model computes the steady state temperature distribution inside the cube for specified boundary conditions

### Custom Boundary Conditions
Each face can be assigned an independent temperature distribution. This is a multivariable function defined over the 2 dimensional region A,

$$
A = \{(x,y)\in\mathbb{R}^2 : -a \leq x \leq a,\ -a \leq y \leq a\}
$$

where a is any real number.


### Three dimensional visualisation
The complete temperature field can be displayed as a colour-mapped three-dimensional scatter plot.

### Interior zoom
Outer layers of the cube can be removed to reveal the internal temperature distribution.

### Cross-Sectional Slices
Temperature slices can be viewed along the x, y or z axes to inspect the interior solution in greater detail.

## Installation (macbook)

Clone the repository:
git clone https://github.com/ad4mh07/heat_transfer_project.git

Install the required packages:
pip install -r requirements.txt

##Usage
Define 6 appropriate functions (can be constant) for the 6 faces, ideally using the naming convention "f_top" etc. Then call the "solve_heat_cube(...)" function from simulation.py . The first 6 inputs are the faces' functions, n represents the resolution, of the cube, ie n=11 creates an 11x11x11 cube (note the cube is still the same size, just the number of nodes has changed). f_bounds states the size of the domain for the inout functions, as described above. zoom (defaulted to 0) allows you 'strip back' layers of outside points, to more easily see internal temperatures. This ability is mutually exclusive to the last 2 parameters, which take an integer (0 to n-1) and 'x', 'y' or 'z', which can be used to view individual slices.
