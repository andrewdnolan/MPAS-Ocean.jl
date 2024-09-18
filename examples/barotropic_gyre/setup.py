#!/usr/bin/env python3

import numpy as np
import xarray as xr
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from polaris.mesh.planar import compute_planar_hex_nx_ny
from polaris.ocean.vertical import init_vertical_coord

def create_culled_mesh(lx, ly, resolution): 
    """

    Parameters
    ----------
    lx : float
        The size of the domain in km in the x direction

    ly : float
        The size of the domain in km in the y direction

    resolution : float
        The resolution of the mesh (distance between cell centers) in km

    """
    dc = resolution * 1e3 # m
    
    nx, ny = compute_planar_hex_nx_ny(lx, ly, resolution)
    ds_mesh = make_planar_hex_mesh(
        nx=nx, ny=ny, dc=dc, nonperiodic_x=True, nonperiodic_y=True
        )
    
    ds_mesh = cull(ds_mesh)
    ds_mesh = convert(ds_mesh)

    return ds_mesh

lx = 1200 # km
ly = 1200 # km

resolution = 100 # km

ds = create_culled_mesh(lx, ly, resolution)

f_0  = 10e-4  # s-1
beta = 10e-11 # s-1 m-1
nLevels = 1 
bottom_depth = 5e3 # m

# set the coriolis
for loc in ["Cell", "Edge", "Vertex"]: 
    ds[f"f{loc}"] = f_0 + beta * ds[f"y{loc}"]

# set the ssh I.C. to zero
ds["ssh"] = xr.zeros_like(ds.xCell).expand_dims(dim='Time', axis=0) 
