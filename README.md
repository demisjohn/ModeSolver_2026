# Purpose

I wanted a unified waveguide mode solver, with a single user interface, but which could execute various freely available python modesolvers.
This electromagnetic waveguide modesolver utilizes the [CAMFR waveguide generation interface](https://github.com/demisjohn/CAMFR#brief-example), but allows the use of the [EMpy modesolvers](https://github.com/lbolla/EMpy) with the same simple interface.

Almost entirely vibe-coded using Cursor & various LLM's in spare time, to solve a specific problem.

## Status

- Really wanted Eigenmode Expansion and/or Field Mode-Matching solvers - didn't really work and I gave up after a bit, as the FEM solver worked ok.  Would prefer EME for very thin layers... another time maybe.
- PML didn't really yield radiating modes and associated optical propagation loss values - perhaps the LLM made some mistake in the math but I didn't really check that hard. "Good enough" for my needs at the time!
