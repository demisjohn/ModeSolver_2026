#!/usr/bin/env python3
"""
Reproduce the rectangular SiO2 / Si3N4 waveguide from the pyFIMM README example
(https://github.com/demisjohn/pyFIMM/blob/master/README.md) and plot the first
five eigenmodes using ModeSolver_2026 (EMpy backend).
"""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

# Allow running without pip install (repo root on path)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ModeSolver_2026 import Material, Slice, Waveguide
import nk   # file `nk.py` in the same directory as this script

#------------------------------------------------------------------------------

wavelength_um = 1.550 # microns

# README indices (dimensionless, as in pyFIMM snippet)
#SiO = Material( nk.SiO2(wavelength_um) )
#SiN = Material( nk.Si3N4(wavelength_um) )
SiO = Material( 1.46 )
SiN = Material( 2.00 )

# 1-D Dimensions (microns), bottom to top
clad = Slice(SiO(2.0))
core = Slice(SiO(2.0 - 0.25/2) + SiN(0.25) + SiO(2.0 - 0.25/2))

# 2-D Waveguide (microns), left to right
WG = Waveguide(clad(3.0) + core(0.5) + clad(3.0))

# plot refractive index profile
fig_n, ax_n = WG.plot(
    title="Refractive index n(x, y) — linear color scale (coolwarm)",
)
out_n = Path(__file__).resolve().parent / "Example1 - RIX profile.png"
fig_n.savefig(out_n)
#plt.close(fig_n)
print(f"Wrote {out_n}")

#WG.plot_RIX()

# 1.55 µm is typical for Si photonics; README leaves wavelength unspecified.
# Default ``solver='SVFD'`` (EMpy semi-vectorial FD) — fast for this demo.
# Use ``solver='VFD'`` or ``solver='vectorial'`` for full EMpy VFDModeSolver (slower).
# Use ``solver='eme'`` for EMEpy's MSEMpy (requires ``pip install emepy``).
WG.calc(
    wavelength_um=wavelength_um,
    neigs=5,
    nx=500,
    ny=500,
    boundary="0000",
    # solver="eme",  # optional: EMEpy MSEMpy (requires ``pip install emepy``)
    # tol=1e-6,
)

print(WG.neff_dataframe().to_string(index=False))

fig, ax = WG.plot("all")

out = Path(__file__).resolve().parent / "Example1 - SiN rect wg modes output.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
