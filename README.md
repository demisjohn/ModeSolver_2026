# Purpose

I wanted a unified waveguide mode solver, with a single user interface, but which could execute various freely available python modesolvers.
This electromagnetic waveguide modesolver utilizes the [CAMFR waveguide generation interface](https://github.com/demisjohn/CAMFR#brief-example), but allows the use of the [EMpy modesolvers](https://github.com/lbolla/EMpy) with the same simple interface.

Almost entirely vibe-coded using Cursor & various LLM's in spare time, to solve a specific problem. ***I have no intention of maintaining this repo*** nor significantly improving it - go ahead and copy/have at it in your own repo!

# Installation

[requirements.txt](https://github.com/demisjohn/ModeSolver_2026/blob/main/requirements.txt) lists the required packages that need to be installed, Cursor somehow knew to use this file to install the required pacakges:
- numpy>=1.20
- scipy>=1.7
- matplotlib>=3.5
- pandas>=1.3
- ElectromagneticPython>=2.0

# Usage

## Brief Example
Example of rectangular waveguide construction syntax: We will create a rectangular waveguide of SiO2 cladding and SiN core, calculate the fundamental mode & plot it. 

Import modules:

    from ModeSolver_2026 import Material, Slice, Waveguide
    import nk   # file `nk.py` contains refractive index info for various materials, by Demis D. John
    
First, create some Materials with some refractive index:

    >>> SiO = Material( 1.45 )    # refractive index of SiO2
    >>> SiN = Material( 2.01 )    # refractive index of Si3N4

(alternatively use my included `nk.py` module which includes refractive indices for common materials.)

Then, create some 1-D slabs, by calling those Materials with a thickness value (in micrometers), and adding them together from bottom to top in a Slice:

    clad = Slice(  SiO(15.75)  )      # Thicknesses in microns
    core = Slice(  SiO(10.0) + SiN(2.5) + SiO(5.0)  )
    
This created an imaginary structure from bottom-to-top. For example `core` looks like:

            top         
    --------------------
            SiO
        5.0 um thick
    --------------------
            SiN
       2.50 um thick
    --------------------
            SiO
       10.0 um thick
    --------------------
           bottom

Then make a 2-D structure by calling these Slices with a width value, and adding them together from left to right in a Waveguide:

    >>> WG = Waveguide(  clad(3.0) + core(1.0) + clad(4.0)  )   # Widths in microns
    
Which creates this imaginary 2-D Waveguide structure from left-to-right:

                                top         
    ---------------------------------------------------------
    |<----- 3.0um------>|<-----1.0um------>|<---- 4.0um---->|
    |                   |        SiO       |                |
    |                   |    5.0 um thick  |                |                
    |                   |------------------|                |
    |        SiO        |        SiN       |       SiO      |
    |      15.75um      |   2.50 um thick  |     15.75um    |
    |       thick       |------------------|      thick     |
    |                   |        SiO       |                |
    |                   |   10.0 um thick  |                |
    ---------------------------------------------------------
                               bottom
    
Plot refractive index ("RIX") profile:

    # plot refractive index profile
    fig_n, ax_n = WG.plot( 
        "RIX",
        title="Refractive index n(x, y)",
    )
    # or: WG.plot_RIX()

![Refractive index profile plot](https://github.com/demisjohn/ModeSolver_2026/blob/main/examples/Example1%20-%20RIX%20profile.png)

Calculate the modes as so:

    # Default ``solver='SVFD'`` (EMpy semi-vectorial FD) — fast for this demo.
    # Use ``solver='VFD'`` or ``solver='vectorial'`` for full EMpy VFDModeSolver (slower).
    WG.calc(
        wavelength_um = 1.55,    # microns
        neigs=5,      # number of modes to calc
        boundary="p",    # options: P (PML), 0 (0-field), see help() for more options.
    )

And plot the modes like so:
    
    >>> WG.plot( 'all' )  # plots all modes on one figure

![Plot of all optical modes](https://github.com/demisjohn/ModeSolver_2026/blob/main/examples/Example1%20-%20SiN%20rect%20wg%20modes%20output.png)

See the Examples directory for full examples, as some details are missing here.



## Status

- See `help(WG)` etc. for documentation - I did check it and pretty sure it's fairly correct, although LLM was not great at consistent docstrings.
- Bend radius is implemented, via setting `WG.bend_radius = 200e-6   # meters` prior to `WG.calc()`.
- Really wanted Eigenmode Expansion and/or Field Mode-Matching solvers - didn't really work with integrating EMEpy's solver, and I gave up after a bit, as the FEM solver worked ok.  Would prefer EME for very thin layers... another time maybe.
- PML didn't really yield radiating modes and associated optical propagation loss values - perhaps the LLM made some mistake in the math but I didn't really check that hard. "Good enough" for my needs at the time!
