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

# Definte Materials w/ refractive indices (RIX)
SiO = Material( nk.SiO2(wavelength_um) )
SiN = Material( nk.Si3N4(wavelength_um) )

# 1-D Dimensions (microns), bottom to top
clad = Slice(SiO(2.0))
core = Slice(SiO(2.0 - 0.25/2) + SiN(0.250) + SiO(2.0 - 0.25/2))

# 2-D Waveguide (microns), left to right
WG = Waveguide(   clad(3.0) + core(0.5) + clad(3.0)   )

# plot refractive index profile
fig_n, ax_n = WG.plot( "RIX",
    title="Refractive index n(x, y)",
)
# or: WG.plot_RIX()

out_n = Path(__file__).resolve().parent / "Example1 - RIX profile.png"
fig_n.savefig(out_n)
#plt.close(fig_n)
print(f"Wrote {out_n}")


# curve the waveguide using index transformation:
WG.bend_radius = 200e-6   # meters 


# plot refractive index profile

fig_n, ax_n = WG.plot( "RIX",
    title="Bend with R = %e mm" % (WG.bend_radius * 1e3),
)
out_n = Path(__file__).resolve().parent / "Example1 - Bend RIX profile.png"
fig_n.savefig(out_n)
#plt.close(fig_n)
print(f"Wrote {out_n}")


# Default ``solver='SVFD'`` (EMpy semi-vectorial FD) — fast for this demo.
# Use ``solver='VFD'`` or ``solver='vectorial'`` for full EMpy VFDModeSolver (slower).
WG.calc(
    wavelength_um=wavelength_um,
    neigs=5,
    boundary="p",    # options: P (PML), 0 (0-field), see help() for more options.
    # tol=1e-6,
)

print(WG)

print("---")
for m in WG.modes:
    print(   m.neff, str(m.get_alpha_dB()) + " dB/m"   )


fig, ax = WG.plot("all", 
                  title = "Bend with R = %e mm" % (WG.bend_radius * 1e3))

out = Path(__file__).resolve().parent / "Example1 - SiN rect wg modes output.png"
fig.savefig(out)
print(f"Wrote {out}")
