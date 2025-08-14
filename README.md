# SimRail

A Python repository for railway wheel-rail contact mechanics simulations.

## Overview

SimRail provides modular tools for modeling and analyzing wheel-rail contact, including both normal and tangential contact problems. The library is organized for extensibility and research use.

## Directory Structure

- `lib/contact_settings.py`  
  Simulation coefficients, constants, and profile paths.

- `lib/contact_material.py`  
  Material properties and friction models.

- `lib/geoLib/contact_wheel_rail.py`  
  Wheel and rail geometry classes.

- `lib/dynLib/contact_dyn_state.py`  
  Dynamic state management for wheel/rail movement.

- `lib/normLib/contact_methods.py`  
  Normal contact models: equivalent ellipse and Kik-Piotrowski.

- `lib/normLib/contact_solver.py`  
  Solver for normal contact patches.

- `lib/tanLib/tangent_methods.py`  
  Tangential contact models (FASTSIM), creepage.

- `calc_routines/determine_static_contact.py`  
  Routines for static contact calculations.

- `example_use.ipynb`  
  Step-by-step usage and visualization notebook.

- `fastsim_example.ipynb`  
  Tangential contact (FASTSIM) demonstration notebook.

## Getting Started

1. **Install dependencies**  
   Run the following commands:
   ```bash
   conda env create -f environment.yml
   conda activate SimRail
   ```

2. **Explore the notebooks**  
   - `example_use.ipynb` for basic workflow and visualization.
   - `fastsim_example.ipynb` for tangential contact calculations.

3. **Typical workflow**
   - Import modules from `lib/`.
   - Initialize wheel, rail, and material objects.
   - Set dynamic states and calculate positions.
   - Solve normal contact using `NormalContactSolver`.
   - Analyze and visualize results.

## Example

See first `example_use.ipynb` for a complete example, including plotting and result analysis. A simple implementation of FASTSIM is done for the equivalent contact in  `fastsim_example.ipynb`.

## License

For research and educational use.
