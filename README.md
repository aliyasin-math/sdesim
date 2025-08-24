# sdesim: Stochastic Differential Equation Simulators

`sdesim` is a Python package for simulating multidimensional ODEs and SDEs with a focus on numerical schemes commonly used in quantitative modeling.  
It is designed both as a reusable library of numerical methods and as an applied toolkit for ecological and biological models.

## Features
- **Numerical integrators**:
  - RK4 (Runge–Kutta 4th order for ODEs)
  - Euler–Maruyama scheme
  - Milstein method
  - Strong order 2 Itô–Taylor scheme
- **Derivative approximations** via high–order finite differences
- **Applications**:
  - Honeybee Hive–plant models (deterministic & stochastic)
  - Disease models (two variants)
  - Pesticide–heat seasonal models

## Installation
Clone the repository and install in editable mode:

```bash
git clone https://github.com/aliyasin-math/sdesim.git
cd sdesim
python3 -m venv .venv
source .venv/bin/activate
pip install -e .