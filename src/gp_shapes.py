import sys
sys.path.append('..')
from jax import vmap
import numpy as np
from src.util import *
import numpy as np

def table_from_data(data_paths):
    tables = []
    for path1 in data_paths:
        row = []
        for path2 in data_paths:
            c1 = jnp.load(path1)
            c2 = jnp.loac(path2)
            row.append(get_dist_table(c1, c2, Nref=100))
        tables.append(row)
    return jnp.vstack([jnp.hstack([c for c in row]) for row in tables])
            

def get_dist_table(clist1, clist2=None, Nref=40):
    if clist2 is None:
        clist2 = clist1
    distance_wrap = lambda x1, x2: distance(x1, x2, Nref=Nref)
    return vmap(vmap(distance_wrap, (0, None)), (None, 0))(clist1, clist2)

def get_align_table(clist1, clist2=None, Nref=40):
    if clist2 is None:
        clist2 = clist1
    rotation_wrap = lambda c1, c2: align_fourier_info(c1, c2, Nref=Nref)
    return vmap(vmap(rotation_wrap, (0, None)), (None, 0))(clist1, clist2)

def get_table_min(table):
    table_mask = jnp.where(jnp.identity(table.shape[0]), jnp.nan, table)
    idx_min = jnp.nanargmin(table_mask)
    (row_min, col_min) = jnp.unravel_index(idx_min, table.shape)
    return row_min, col_min


def get_table_max(table):
    table_mask = jnp.where(jnp.identity(table.shape[0]), jnp.nan, table)
    idx_max = jnp.nanargmax(table_mask)
    (row_max, col_max) = jnp.unravel_index(idx_max, table.shape)
    return row_max, col_max