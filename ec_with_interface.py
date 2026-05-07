import numpy as np
import os
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix

from copy import deepcopy
from pathlib import Path

from skimage import measure
import trimesh
from trimesh.smoothing import laplacian_calculation

import gemmi
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from mesh import find_neighbor_indexes, make_interface_grid, calculate_smooth_sdf, SurfaceMesh


PROTEIN_CUTOFF = 15.0
DISTANCE_THRESHOLD = 6.0   # which atoms belong to the interface
GRID_PADDING = 2.0                   # Å, extra space around the interface
GRID_POINTS_PER_ANGSTROM = 1         # grid resolution
SDF_CUTOFF_RADIUS = 5.0              # max distance for smooth sum
SDF_SMOOTH_FACTOR = 1.5              # exponent sharpness


def parse_pqr(pqr_file: str) -> tuple[np.array, np.array, np.array]:
    """
    Извлечение координат, зарядов и радиусов атомов из pqr-файла
    """
    coords, charges, radii = [], [], []
    with open(pqr_file, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                parts = line.split()
                radii.append(float(parts[-1]))
                charges.append(float(parts[-2]))
                coords.append([float(parts[-5]), float(parts[-4]), float(parts[-3])])

    return np.array(coords), np.array(charges), np.array(radii)


def calc_esp(
        atom_coords: np.ndarray,
        atom_charges: np.ndarray,
        surface_coords: np.ndarray
) -> np.array:
    """
    Вычисление относительного электростатического потенциала (без учета 1/4pi*e0 из-за корреляции Пирсона).
    """
    esp_values = []
    tree = cKDTree(atom_coords)
    surf_tree = cKDTree(surface_coords)
    sparse_dist = surf_tree.sparse_distance_matrix(tree, PROTEIN_CUTOFF, output_type='coo_matrix')
    distances = sparse_dist.data
    rows = sparse_dist.row
    cols = sparse_dist.col

    relevant_charges = atom_charges[cols]
    partial_potentials = relevant_charges / (distances + 1e-10)
    potentials = np.bincount(rows, weights=partial_potentials, minlength=len(surface_coords))

    return potentials


def main(lig_pqr, prot_pqr):
    l_data = parse_pqr(lig_pqr)
    p_data = parse_pqr(prot_pqr)
    if not l_data or not p_data: return None

    l_coords, l_charges, l_radii = l_data
    p_coords, p_charges, p_radii = p_data

    # 2. Генерация гладкого интерфейса
    print(f"   - Computing smooth SDF interface...")
    s_sdf, grd_xx, grd_yy, grd_zz = calculate_smooth_sdf(
        l_coords, l_radii,
        p_coords, p_radii,
        distance_threshold=DISTANCE_THRESHOLD,
        grid_padding=GRID_PADDING,
        grid_points_per_angstrom=GRID_POINTS_PER_ANGSTROM,
        cutoff_radius=SDF_CUTOFF_RADIUS,
        smooth_factor=SDF_SMOOTH_FACTOR
    )

    # Получение изоповерхности
    print(f"   - Creating smooth isosurface...")
    mesh = SurfaceMesh.from_sdf(s_sdf, grd_xx, grd_yy, grd_zz)

    # 4. Расчет ESP
    print(f"   - Calculating ESP values...")
    surf_points = mesh.vertices
    v_lig = calc_esp(l_coords, l_charges, surf_points)
    v_prot = calc_esp(p_coords, p_charges, surf_points)

    # 5. Корреляция
    print(f"   - Calculating Pearson correlation...")
    corr, _ = pearsonr(v_lig, v_prot)
    print(corr)

    print(f"   - Calculating local EC values...")
    EC = v_lig + v_prot
    print(EC)

    return v_lig, v_prot, corr, EC


if __name__ == "__main__":
    main('lig1_charged.pqr', '5c7a_charged.pqr')
