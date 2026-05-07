import numpy as np
from scipy.spatial import cKDTree

from skimage import measure
import trimesh


PROTEIN_CUTOFF = 15.0
DISTANCE_THRESHOLD = 6.0   # which atoms belong to the interface
GRID_PADDING = 2.0                   # Å, extra space around the interface
GRID_POINTS_PER_ANGSTROM = 1.0       # grid resolution
SDF_CUTOFF_RADIUS = 5.0              # max distance for smooth sum
SDF_SMOOTH_FACTOR = 1.5              # exponent sharpness


class SurfaceMesh(trimesh.base.Trimesh):
    """
        Класс для удобного хранения меша,
        чтобы не таскать несколько массивов
    """

    def __init__(self, vertices: np.ndarray, faces, normales, values=None):
        if values is None:
            values = np.zeros(vertices.shape[0])
        super().__init__(
            vertices=vertices,
            faces=faces,
            face_normals=normales,
            vertex_attributes=dict(
                values=values
            )
        )

    @property
    def values(self):
        return self.vertex_attributes.get("values")

    @classmethod
    def from_sdf(
            cls: "SurfaceMesh",
            sdf: np.ndarray,
            grd_xx: np.ndarray,
            grd_yy: np.ndarray,
            grd_zz: np.ndarray,
    ) -> "SurfaceMesh":
        # Вычисление начала координат сетки и spacing
        origin = np.asarray([
            grd_xx[0, 0, 0],
            grd_yy[0, 0, 0],
            grd_zz[0, 0, 0],
        ])
        spacing = (
            grd_xx[1, 0, 0] - grd_xx[0, 0, 0],
            grd_yy[0, 1, 0] - grd_yy[0, 0, 0],
            grd_zz[0, 0, 1] - grd_zz[0, 0, 0],
        )

        # Получение меша
        verts, faces, normals, values = measure.marching_cubes(sdf, level=0, spacing=spacing)

        # Сдвиг вершин на origin
        verts_physical = verts + origin

        # Обработка нормалей
        # Нормали уже в физическом пространстве, но могут нуждаться в перенормировке
        # если spacing анизотропный (разные в разных направлениях)
        normals_physical = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        return cls(verts_physical, faces, normals_physical)


def find_neighbor_indexes(a_xyz, b_xyz,
                          distance_threshold=DISTANCE_THRESHOLD):
    a_tree = cKDTree(a_xyz)
    b_tree = cKDTree(b_xyz)

    a_neighbors_idx = list({nbr for nbr_list in a_tree.query_ball_point(b_xyz, distance_threshold)
                            for nbr in nbr_list})
    b_neighbors_idx = list({nbr for nbr_list in b_tree.query_ball_point(a_xyz, distance_threshold)
                            for nbr in nbr_list})
    return a_neighbors_idx, b_neighbors_idx


def make_interface_grid(a_xyz, b_xyz,
                        distance_threshold=DISTANCE_THRESHOLD,
                        grid_padding=GRID_PADDING,
                        grid_points_per_angstrom=GRID_POINTS_PER_ANGSTROM):
    a_neighbors_idx, b_neighbors_idx = find_neighbor_indexes(a_xyz, b_xyz, distance_threshold)
    interface_atoms = np.concatenate([a_xyz[a_neighbors_idx],
                                      b_xyz[b_neighbors_idx]], axis=0)

    limits = [(np.min(interface_atoms.T[i]), np.max(interface_atoms.T[i]))
              for i in range(3)]

    grd_xx, grd_yy, grd_zz = np.meshgrid(
        *(np.linspace(d_min - grid_padding,
                      d_max + grid_padding,
                      grid_points_per_angstrom * np.ceil(d_max - d_min + 2 * grid_padding).astype(int))
          for d_min, d_max in limits),
        indexing='ij')
    return grd_xx, grd_yy, grd_zz


def calculate_smooth_sdf(
        a_xyz: np.ndarray,
        a_vdw: np.ndarray,
        b_xyz: np.ndarray,
        b_vdw: np.ndarray,
        *,
        distance_threshold: float = 6.0,
        grid_padding: float = 0.0,
        grid_points_per_angstrom: int = 1,
        cutoff_radius: float = 5.0,
        smooth_factor: float = 1.5,
) -> np.ndarray:

    # вычисляем сетку
    grd_xx, grd_yy, grd_zz = make_interface_grid(a_xyz, b_xyz, distance_threshold, grid_padding,
                                                 grid_points_per_angstrom)

    # Собираем координаты узлов сетки в плоский массив точек [N*M*K, 3]
    grid_points = np.c_[
        grd_xx.ravel(),
        grd_yy.ravel(),
        grd_zz.ravel(),
    ]
    grid_tree = cKDTree(grid_points)

    # Храним координаты в виде KD-деревьев
    a_tree = cKDTree(a_xyz)
    b_tree = cKDTree(b_xyz)

    # Вычисляем расстояние до ближайших к точке сетки атомов и их индексы
    # Работаем с разреженной матрицей расстояний (формат COO)
    # i — индексы grid_xyz, j — индексы атомов
    a_sparse_dist = grid_tree.sparse_distance_matrix(a_tree, cutoff_radius, output_type='coo_matrix')
    # sparse_dist.data — это уже вычисленные расстояния d = |G - P|
    # sparse_dist.row — индексы точек сетки (i)
    # sparse_dist.col — индексы атомов (j)
    a_d_squared = np.power(a_sparse_dist.data, 2)
    a_vdw_squared = np.power(a_vdw[a_sparse_dist.col], 2)  # сопоставляем vdw по индексам
    a_exp_values = np.exp(-smooth_factor * (a_d_squared - a_vdw_squared))
    # Агрегируем значения обратно по строкам (по точкам сетки)
    # Используем np.bincount, это быстрее, чем преобразование в CSR и sum(axis=1)
    a_grid_sums = np.bincount(a_sparse_dist.row, weights=a_exp_values, minlength=len(grid_points))

    b_sparse_dist = grid_tree.sparse_distance_matrix(b_tree, cutoff_radius, output_type='coo_matrix')
    b_d_squared = np.power(b_sparse_dist.data, 2)
    b_vdw_squared = np.power(b_vdw[b_sparse_dist.col], 2)  # сопоставляем vdw по индексам
    b_exp_values = np.exp(-smooth_factor * (b_d_squared - b_vdw_squared))
    b_grid_sums = np.bincount(b_sparse_dist.row, weights=b_exp_values, minlength=len(grid_points))

    # Считаем SDF
    sdf_points = np.log(a_grid_sums + 1e-5) - np.log(b_grid_sums + 1e-5)
    sdf_tensor = sdf_points.reshape(grd_xx.shape)

    return sdf_tensor, grd_xx, grd_yy, grd_zz