import numpy as np
import subprocess
import os
from scipy.stats import pearsonr
import csv


DENSITY = 1.0        # точек на Ангстрем^2
PROBE_RADIUS = 1.4   # радиус зонда воды (SAS)
PROTEIN_CUTOFF = 15.0   # обрезка белка вокруг лиганда


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


def distribute_on_sphere(
        n: int,
        r: int
) -> np.column_stack:
    """
    Генерация координат по Фибоначчи для равномерного распределния
    (1 точка на 1 ангстрем^2)
    """

    # Создаем массив индексов от 0 до n
    i = np.arange(0, n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden_ratio = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / golden_ratio

    # Сферические координаты в декартовы
    x = r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(phi)

    return np.column_stack((x, y, z))


def generate_surface(
        coords: np.array,
        radii: np.array,
        probe=PROBE_RADIUS,
        density=DENSITY
) -> np.vstack:

    """
    Генерация поверхности
    """
    surface_points = []
    effective_radii = radii + probe

    for i, center in enumerate(coords):
        # 1. Генерируем точки на сфере вокруг текущего атома
        R = effective_radii[i]
        dots = distribute_on_sphere(int(4 * np.pi * R ** 2 * density), R) + center

        # 2. Фильтрация: точка должна быть снаружи всех остальных сфер
        keep = np.ones(len(dots), dtype=bool)

        for j, other_center in enumerate(coords):
            if i == j: continue

            dist_sq = np.sum((dots - other_center) ** 2, axis=1)
            keep &= dist_sq >= (effective_radii[j] ** 2)

        surface_points.append(dots[keep])

    return np.vstack(surface_points)


def calc_esp(
        atom_coords: np.array,
        atom_charges: np.array,
        surface_coords: np.vstack
) -> np.array:
    """
    Вычисление относительного электростатического потенциала (без учета 1/4pi*e0 из-за корреляции Пирсона).
    """
    esp_values = []

    for p in surface_coords:
        # norm means sqrt((xp - xi) ** 2 + (yp - yi) ** 2 + (yp - yi) ** 2)
        distances = np.linalg.norm(atom_coords - p, axis=1)
        distances = np.where(distances == 0, 1e-10, distances)
        potential = np.sum(atom_charges / distances)    # коэффициенты?
        esp_values.append(potential)

    return np.array(esp_values)


def main(lig_pqr, prot_pqr):
    l_data = parse_pqr(lig_pqr)
    p_data = parse_pqr(prot_pqr)
    if not l_data or not p_data: return None

    l_coords, l_charges, l_radii = l_data
    p_coords, p_charges, _ = p_data

    # 2. Генерация поверхности лиганда
    print(f"   - Generating surface for {os.path.basename(lig_pqr)}...")
    surf_points = generate_surface(l_coords, l_radii)

    # 3. Отсечение активного центра белка (cutoff)
    lig_center = np.mean(l_coords, axis=0)
    p_dists = np.linalg.norm(p_coords - lig_center, axis=1)
    active_p_mask = p_dists <= PROTEIN_CUTOFF
    p_coords_act = p_coords[active_p_mask]
    p_charges_act = p_charges[active_p_mask]

    # 4. Расчет ESP
    print(f"   - Calculating ESP values...")
    v_lig = calc_esp(l_coords, l_charges, surf_points)
    v_prot = calc_esp(p_coords_act, p_charges_act, surf_points)

    # 5. Корреляция
    corr, _ = pearsonr(v_lig, v_prot)
    print(corr)

    return surf_points, v_lig, v_prot, corr


if __name__ == "__main__":
    main('lig1_charged.pqr', '5c7a_charged.pqr')
