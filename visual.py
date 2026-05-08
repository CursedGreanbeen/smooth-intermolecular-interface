import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
from mesh import find_neighbor_indexes


def visualizer(lig_xyz, prot_xyz, lig_vdw, prot_vdw, smooth_mesh):
    lig_neighbors_idx, prot_neighbors_idx = find_neighbor_indexes(
        lig_xyz,
        prot_xyz,
        6.0,
    )

    prot_full = go.Scatter3d(
        x=prot_xyz.T[0],
        y=prot_xyz.T[1],
        z=prot_xyz.T[2],
        marker=go.scatter3d.Marker(
            color="#CD9EF7",
            size=prot_vdw * 3,
        ),
        mode="markers",
        name="Protein",
    )
    lig_full = go.Scatter3d(
        x=lig_xyz.T[0],
        y=lig_xyz.T[1],
        z=lig_xyz.T[2],
        marker=go.scatter3d.Marker(
            color="#64BAFF",
            size=lig_vdw * 3,
        ),
        mode="markers",
        name="Ligand",
    )
    prot_interface = go.Scatter3d(
        x=prot_xyz.T[0, prot_neighbors_idx],
        y=prot_xyz.T[1, prot_neighbors_idx],
        z=prot_xyz.T[2, prot_neighbors_idx],
        marker=go.scatter3d.Marker(
            color="#8A56B9",
            size=prot_vdw * 5,
            opacity=0.5,
        ),
        mode="markers",
        name="Protein Interface",
    )
    lig_interface = go.Scatter3d(
        x=lig_xyz.T[0, lig_neighbors_idx],
        y=lig_xyz.T[1, lig_neighbors_idx],
        z=lig_xyz.T[2, lig_neighbors_idx],
        marker=go.scatter3d.Marker(
            color="#317AB5",
            size=lig_vdw * 5,
            opacity=0.5,
        ),
        mode="markers",
        name="Ligand Interface",
    )

    # Получаем максимальное значение ЕС
    EC = np.abs(smooth_mesh.vertex_attributes['EC'])
    M = max(abs(EC)) if len(EC) > 0 else 1.0

    interface = go.Mesh3d(
        x=smooth_mesh.vertices.T[0],
        y=smooth_mesh.vertices.T[1],
        z=smooth_mesh.vertices.T[2],
        i=smooth_mesh.faces.T[0],
        j=smooth_mesh.faces.T[1],
        k=smooth_mesh.faces.T[2],
        opacity=0.5,
        flatshading=True,

        # Раскраска по локальным значениям ЕС
        intensity=EC,
        colorscale=[[0, 'green'], [0.5, 'white'], [1, 'red']],
        cmin=0,
        cmax=M,
        showscale=True,
        name="Interface EC",
    )

    x_edges, y_edges, z_edges = [], [], []
    for face in smooth_mesh.faces:
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            x_edges.extend([smooth_mesh.vertices[v0, 0], smooth_mesh.vertices[v1, 0], None])
            y_edges.extend([smooth_mesh.vertices[v0, 1], smooth_mesh.vertices[v1, 1], None])
            z_edges.extend([smooth_mesh.vertices[v0, 2], smooth_mesh.vertices[v1, 2], None])

    # grid = go.Scatter3d(
    #     x=x_edges, y=y_edges, z=z_edges,
    #     mode='lines',
    #     line=dict(color='#FFFFFF', width=1),
    #     hoverinfo='skip',
    #     name="Interface Mesh",
    # )

    fig = go.Figure(
        data=[
            prot_full,
            lig_full,
            prot_interface,
            lig_interface,
            interface,
            # grid,

        ],
    )

    fig.update_layout(
        width=1200,
        height=900,
        scene=dict(
            xaxis=dict(range=[lig_xyz.T[0].min() - 10, lig_xyz.T[0].max() + 10]),
            yaxis=dict(range=[lig_xyz.T[1].min() - 10, lig_xyz.T[1].max() + 10]),
            zaxis=dict(range=[lig_xyz.T[2].min() - 10, lig_xyz.T[2].max() + 10]),
            aspectmode='data'  # Кубические пропорции — иногда помогает
        ),
        legend=dict(
            x=0.02,  # place legend near left edge
            y=0.98,  # near top edge
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.7)'  # semi-transparent background for readability
        ),
        margin=dict(r=120)  # extra space on right for the colorbar
    )

    fig.show()


def export_to_pymol(
        sdf: np.ndarray,
        grd_xx: np.ndarray,
        grd_yy: np.ndarray,
        grd_zz: np.ndarray,
        file: Path,
):
    """
    Export scalar grid to PyMOL-compatible OpenDX format.

    Parameters
    ----------
    grid : np.ndarray of shape (nx, ny, nz)
        Your scalar density field
    origin : tuple (x, y, z)
        Physical coordinates of grid point (0, 0, 0)
    spacing : tuple (dx, dy, dz)
        Grid spacing in Angstroms (or any unit PyMOL uses)
    """
    # Вычисление начала координат сетки и spacing
    origin = (
        grd_xx[0, 0, 0],
        grd_yy[0, 0, 0],
        grd_zz[0, 0, 0],
    )
    spacing = (
        grd_xx[1, 0, 0] - grd_xx[0, 0, 0],
        grd_yy[0, 1, 0] - grd_yy[0, 0, 0],
        grd_zz[0, 0, 1] - grd_zz[0, 0, 0],
    )
    nx, ny, nz = sdf.shape

    file.parent.mkdir(exist_ok=True, parents=True)

    with file.open("w") as f:
        # Header defining the grid geometry
        f.write(f'object 1 class gridpositions counts {nx} {ny} {nz}\n')
        f.write(f'origin {origin[0]} {origin[1]} {origin[2]}\n')
        f.write(f'delta {spacing[0]} 0 0\n')
        f.write(f'delta 0 {spacing[1]} 0\n')
        f.write(f'delta 0 0 {spacing[2]}\n')
        f.write(f'object 2 class gridconnections counts {nx} {ny} {nz}\n')
        f.write(f'object 3 class array type float rank 0 items {nx * ny * nz} data follows\n')

        # Write data (Fortran order: x varies fastest)
        # This matches OpenDX expectation: grid[i,j,k] with i (x-index) changing fastest
        for val in sdf.ravel(order='C'):
            f.write(f'{val:.6e}\n')

        f.write(f'object "density" class field\n')
