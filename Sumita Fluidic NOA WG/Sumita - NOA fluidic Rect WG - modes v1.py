#!/usr/bin/env python3
"""
Simulate modes in a waveguide utilizing 
Norland optical adhesive (NOA-61) as the core along with a water fluidic channel.

Demis D. John
collab w/ Sumita Pennathur
Univ. California, Santa Barbara, original code: 2026-04-11

--------------------------------

Requires: (somewhere in the accessible PATH)
nk.py : https://github.com/demisjohn/nk.py
ModeSolver_2026 : https://github.com/demisjohn/ModeSolver_2026
EMpy : https://github.com/lbolla/EMpy

"""

from pathlib import Path
from datetime import datetime
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

# Allow running without pip install (repo root on path)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt

from ModeSolver_2026 import Material, Slice, Waveguide
import nk   # file `nk.py` in the same directory as this script

#------------------------------------------------------------------------------

fileout = "NOA fluidic Rect WG - modes" 
wavelength_um = 0.480 # microns

# Materials: refractive indices 
SiO = Material( nk.SiO2(wavelength_um) )
NOA = Material( 1.56 )  # Norland Optical Adhesive (NOA-61 or 88)
Water = Material( 1.333 ) # Water
Air = Material( 1.0 ) # Air

sim_height = 5.0 # microns, simulation height
sim_width = 10.0 # microns, simulation width

# 1-D Dimensions (microns) - vertical concat from bottom-to-top
clad =     Slice(    SiO(2.0 + 0.100 + 0.200)  + NOA(0.100) + SiO(sim_height - (2+0.1+.1+.2+.1) )  )
core_outer = Slice(    SiO(2.0) + Water(0.100 + 0.200) + NOA(0.100) + SiO(sim_height - (2+0.3+0.1) )   )
core_mid =  Slice(    SiO(2.0) + NOA(0.100) + Water(0.200) + NOA(0.100) + SiO(sim_height - (2+0.1+0.2+0.1) )   )

# 2-D Waveguide (microns) - horizontal concat from left-to-right
WG = Waveguide(clad(3.0) + core_outer(0.5) + core_mid(1.0) + core_outer(0.5) + clad(3.0))

fig_rix, ax_rix = WG.plot_refractive_index_profile(WG)
fig_rix.show()

filedate = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]}"
out = Path(__file__).resolve().parent / f"{fileout + ' - RIX - ' + filedate}.png"
fig_rix.savefig(out, dpi=150)
print(f"Wrote {out}")
#plt.close(fig_rix)

# 1.55 µm is typical for Si photonics; README leaves wavelength unspecified.
# Scalar finite-difference solver (EMpy SVFD) — fast and adequate for this demo.
# Use WG.calc(..., vectorial=True) for full-vectorial VFDModeSolver (much slower).
print("Calculating modes...")
WG.calc(wavelength_um=1.55, neigs=5, nx=500, ny=500, boundary="0000")

print(WG.neff_dataframe().to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
axes_flat = axes.ravel()
for i in range(5):
    m = WG.mode(i)
    m.plot_intensity(ax=axes_flat[i], title=f"Mode {i}, neff = {m.neff.real:.4f}")
axes_flat[5].axis("off")
fig.suptitle(
    fileout,
    fontsize=11,
)


out = Path(__file__).resolve().parent / f"{fileout + ' - Modes - ' + filedate}.png"
fig.savefig(out, dpi=150)
print(f"Wrote {out}")
