# Copyright 2023 Euratom
# Copyright 2023 United Kingdom Atomic Energy Authority
# Copyright 2023 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.
"""Module for loading wall components from wall IDSs."""

import numpy as np
from raysect.core.math.polygon import triangulate2d

from imas.ids_structure import IDSStructure

__all__ = ["load_wall_3d"]

VERTEX_DIMENSION = 0
POLYGON_DIMENSION = 2


# TODO: Check coordinate types and convert to Cartesian if required
def load_wall_3d(
    description_ggd: IDSStructure, subsets: list[str] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Load machine wall components from IMAS wall IDS.

    Parameters
    ----------
    description_ggd
        A description_ggd structure from the 'wall' IDS.
    subsets
        List of names of specific ggd subsets to load, by default None (loads all subsets).

    Returns
    -------
    dict[str, dict[str, np.ndarray]]
        Dictionary of wall components defined by vertices and triangles.
        The dictionary keys for components are assigns as follows:
        ``"{grid_name}.{subset_name}.{material_name}"``
        E.g.: ``"FullTokamak.full_main_chamber_wall.Be"``.

    Raises
    ------
    RuntimeError
        If the grid_ggd AOS is empty in the given description_ggd.
    """
    if not len(description_ggd.grid_ggd):
        raise RuntimeError("The grid_ggd AOS is empty in the given description_ggd.")

    # get the grid
    grid = description_ggd.grid_ggd[0]

    # get the material names and the element indices of grid subsets
    material_struct = description_ggd.material
    materials = _get_materials(material_struct)

    # get the coordinates of the vertices
    space = grid.space[0]  # wall grid has only a single space (unstructured grid)
    num_vert = len(space.objects_per_dimension[VERTEX_DIMENSION].object)
    vertices = np.zeros((num_vert, 3))
    for i in range(num_vert):
        vertices[i] = space.objects_per_dimension[VERTEX_DIMENSION].object[i].geometry

    poly_in_subsets = set()  # the polygons in grid subsets
    components = {}

    grid_name = grid.identifier.name

    # iterate over the grid subsets to get individual wall components and their materials
    for subset in grid.grid_subset:
        subset_name = subset.identifier.name
        subset_index = subset.identifier.index

        if subset.dimension - 1 != POLYGON_DIMENSION:
            continue
        if subsets and (subset_name not in subsets):
            continue

        if subset_index in materials:
            # materials are specified for this subset
            for material_name, element_indices in materials[subset_index]:
                component_name = f"{grid_name}.{subset_name}.{material_name}"
                vert, tri = _get_mesh_from_subset(
                    grid, subset, element_indices, vertices, poly_in_subsets
                )
                components[component_name] = {"vertices": vert, "triangles": tri}
        else:
            # materials are not specified
            component_name = f"{grid_name}.{subset_name}.none"
            element_indices = range(len(subset.element)) if len(subset.element) else [0]
            vert, tri = _get_mesh_from_subset(
                grid, subset, element_indices, vertices, poly_in_subsets
            )
            components[component_name] = {"vertices": vert, "triangles": tri}

    # add all remaining polygons to a dedicated component
    if not subsets:
        vert, tri = _get_mesh_from_remaining_polygons(grid, vertices, poly_in_subsets)
        component_name = f"{grid_name}.none.none"
        components[component_name] = {"vertices": vert, "triangles": tri}

    return components


def _get_materials(material_struct):
    materials = {}

    if len(material_struct):  # check if the material structure is defined in this IDS
        for subset in material_struct[0].grid_subset:
            names = np.asarray(subset.identifiers.names)
            # get unique materials
            unique_names = np.unique(names)
            material_list = []
            for name in unique_names:
                (elements,) = np.where(names == name)
                material_list.append((name, elements))
            materials[subset.grid_subset_index] = material_list

    return materials


def _get_mesh_from_subset(grid, subset, element_indices, vertices, poly_in_subsets):
    space = grid.space[0]  # wall grid has only a single space (unstructured grid)
    triangles = []

    if len(subset.element):  # complex subset
        indices = [
            subset.element[iel].object[0].index - 1 for iel in element_indices
        ]  # indexing: 1 -> 0
    else:  # trivial subset, includes all polygons, element index equals to polygon index
        if len(element_indices) == 1 and element_indices[0] == 0:  # trivial case, use all polygons
            indices = range(len(space.objects_per_dimension[POLYGON_DIMENSION].object))
        else:
            indices = element_indices

    for i in indices:
        polygon = (
            space.objects_per_dimension[POLYGON_DIMENSION].object[i].nodes - 1
        )  # indexing: 1 -> 0
        if len(polygon) == 3:
            triangles.append(polygon)
        elif len(polygon) > 3:
            tri_array = polygon[_triangulate_polygon(vertices[polygon])]
            triangles += [tri for tri in tri_array]
        else:
            raise RuntimeError(f"Not a polygon: {np.array2string(polygon)}.")

    poly_in_subsets.update(indices)

    triangles = np.array(triangles, dtype=np.int32)
    vert_index, inv_index = np.unique(triangles, return_inverse=True)
    vert = vertices[vert_index]  # vertices in this subset
    tri = np.arange(len(vert), dtype=np.int32)[inv_index].reshape(triangles.shape)  # renumerate

    return vert, tri


def _get_mesh_from_remaining_polygons(grid, vertices, poly_in_subsets):
    space = grid.space[0]  # wall grid has only a single space (unstructured grid)
    num_poly = len(space.objects_per_dimension[POLYGON_DIMENSION].object)
    mask = np.ones(num_poly, dtype=bool)
    mask[list(poly_in_subsets)] = False
    (poly_remain,) = np.where(mask)

    triangles = []
    for i in poly_remain:
        polygon = (
            space.objects_per_dimension[POLYGON_DIMENSION].object[i].nodes - 1
        )  # indexing: 1 -> 0
        if len(polygon) == 3:
            triangles.append(polygon)
        elif len(polygon) > 3:
            tri_array = polygon[_triangulate_polygon(vertices[polygon])]
            triangles += [tri for tri in tri_array]
        else:
            raise RuntimeError(f"Not a polygon: {np.array2string(polygon)}.")

    triangles = np.array(triangles, dtype=np.int32)
    vert_index, inv_index = np.unique(triangles, return_inverse=True)
    vert = vertices[vert_index]  # vertices in this subset
    tri = np.arange(len(vert), dtype=np.int32)[inv_index].reshape(triangles.shape)  # renumerate

    return vert, tri


def _triangulate_polygon(vert):
    # convert to 2d coordinates
    vert -= vert[0]
    e1 = np.copy(vert[1])
    e2 = np.copy(vert[2])
    e3 = np.cross(e1, e2)
    e3 /= np.sqrt(e3 @ e3)
    e1 /= np.sqrt(e1 @ e1)
    e2 = np.cross(e3, e1)
    transform = np.linalg.inv(np.vstack((e1, e2, e3)))

    polygon2d = (vert @ transform)[:, :2]

    triangles = triangulate2d(polygon2d)

    return triangles[:, ::-1]  # clockwise -> anti-clockwise
