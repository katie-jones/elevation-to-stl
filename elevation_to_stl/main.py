#!/usr/bin/env python3

# Copyright (2020) Katie Jones. All rights reserved.
"""Convert elevation data to STL file"""

import argparse
import elevation
import itertools
import numpy as np
import os
import rasterio
import requests
import srtm
import sys
import yaml
import zlib

from stl import mesh

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Approximate factors to convert latitude/longitude to meters.
LONGITUDE_DEG_TO_METERS = 111111 * np.cos(np.sqrt(np.pi / 4))
LATITUDE_DEG_TO_METERS = 111111

# Factor by which to scale Z values to make them more visible.
Z_SCALE = 2

# Offset between minimum Z point and bottom surface [m].
BOTTOM_SURFACE_OFFSET = 100


class ElevationReader(object):
    """Read elevation data."""
    def get_elevation(self, lat, lon):
        """Get elevation at given latitude and longitude.

        Inputs:
            lat: Latitude (North) at which to get elevation.
            lon: Longitude (East) at which to get elevation.

        Outputs:
            elev: Elevation [m].
        """
        raise NotImplementedError(
            'This function is not implemented in the parent class.')


class SRTMReader(ElevationReader):
    """Read SRTM data from server."""

    # Number of entries in each row/column of each dataset.
    DATASET_DIMENSION = 3601

    # Max number of cache entries.
    N_CACHE_ENTRIES = 4

    # Base URL for requests.
    BASE_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/skadi'

    def __init__(self):
        """Initialize cache to be empty."""
        self.cache = []
        self.cache_metadata = []
        self.fs_cache_location = os.path.join('data', 'SRTM1')

    def _get_elevation_without_interpolation(self, lat, lon):
        """Get elevation data from nearest available point (no interpolation).

        Inputs:
            lat: Latitude (North) for which to get nearest elevation data.
            lon: Longitude (East) for which to get nearest elevation data.

        Outputs:
            (elev, (lat, lon))
            elev: Elevation [m].
            lat: Latitude at which elevation was evaluated.
            lon: Longitude at which elevation was evaluated.
        """
        # Get primary dataset (with row and column) where we can find the
        # nearest value.
        lat_deg_floor = int(np.floor(lat))
        lon_deg_floor = int(np.floor(lon))
        dataset = self._get_dataset_name(lat_deg_floor, lon_deg_floor)
        row = int(
            np.round((lat_deg_floor + 1 - lat) * (self.DATASET_DIMENSION - 1)))
        col = int(
            np.round((lon - lon_deg_floor) * (self.DATASET_DIMENSION - 1)))

        indices = [(dataset, (row, col))]

        # Get secondary datasets (with row and column).
        # These are possible because the first and last row and column of each
        # dataset is duplicated in the neighbouring datasets.
        # TODO: Add these datasets.
        secondary_indices = []

        indices.extend(secondary_indices)

        # Read elevation data from cache.
        elevation = None
        for dataset, (row, col) in indices:
            elevation = self._read_from_cache(dataset, row, col)
            if elevation is not None:
                break

        # If not found in cache, read first dataset from file into cache and
        # get elevation.
        if elevation is None:
            dataset, (row, col) = indices[0]
            self._load_to_cache(dataset)
            elevation = self._read_from_cache(dataset, row, col)

        return elevation

    def _read_from_cache(self, dataset, row, col):
        """Read data from cache and return None if not found.

        Inputs:
            dataset: Name of dataset to read from
            row: Row of data to read
            col: Column of data to read

        Outputs:
            data: Data read from cache. None if value not found.
        """
        idx_search = [
            i for i, e in enumerate(self.cache_metadata)
            if e['dataset'] == dataset
        ]
        return None if not idx_search else self.cache[idx_search[0]][row, col]

    def _load_to_cache(self, dataset):
        """Load the given dataset from file or server to cache.

        Loads the dataset from file if the file exists, else queries the server 
        and saves dataset to file before reading.

        Inputs:
            dataset: Name of dataset to load

        Outputs:
            None
        """
        # Load from file (from server if file does not exist).
        fname = os.path.join(self.fs_cache_location, dataset + '.hgt')

        def load_data_from_file(fname):
            """Load data from file, checking file size."""
            dim = self.DATASET_DIMENSION
            if (os.path.getsize(fname) != dim * dim * 2):
                raise RuntimeError(
                    f'File {fname:s} has wrong size and may be corrupt.')
            return np.fromfile(fname, np.dtype('>i2'), dim * dim).reshape(
                (dim, dim))

        try:
            data = load_data_from_file(fname)
        except:
            response = requests.get(self._get_dataset_url(dataset))
            raw = zlib.decompress(response.content, zlib.MAX_WBITS | 32)
            with open(fname, 'wb') as f:
                f.write(raw)

            data = load_data_from_file(fname)

        # Put data in cache.
        if len(self.cache) < self.N_CACHE_ENTRIES:
            self.cache.append(data)
            self.cache_metadata.append({
                'dataset': dataset,
                'last_used': -1,
            })

        else:
            idx = min(enumerate(self.cache_metadata),
                      key=lambda e: e[1]['last_used'])[0]
            self.cache[idx] = data
            self.cache_metadata[idx] = {
                'dataset': dataset,
                'last_used': -1,
            }

    @staticmethod
    def _get_dataset_name(lat, lon):
        """Get dataset name from longitude and latitude."""
        return f'N{lat:02d}E{lon:03d}'

    @classmethod
    def _get_dataset_url(cls, dataset):
        """Get URL for request for dataset."""
        return '/'.join([cls.BASE_URL, dataset[:3], dataset + '.hgt.gz'])

    def get_elevation(self, lat, lon):
        """Get elevation at given latitude and longitude.

        Inputs:
            lat: Latitude (North) at which to get elevation.
            lon: Longitude (East) at which to get elevation.

        Outputs:
            elev: Elevation [m].
        """
        return self._get_elevation_without_interpolation(lat, lon)


def parse_arguments(argv):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser('elevation-to-stl')
    parser.add_argument('config_file',
                        help='Filename of input configuration file')
    parser.add_argument('output_file', help='Filename of output STL file')
    return parser.parse_args(argv)


def main(argv):
    """Generate STL file from elevation data."""
    # Get configuration.
    args = parse_arguments(argv[1:])
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Get data from given range.
    n = 100
    reader = SRTMReader()

    lonvec = np.linspace(*config['longitude'], n)
    latvec = np.linspace(*config['latitude'], n)
    longrid, latgrid = np.meshgrid(lonvec, latvec)
    elevgrid = np.empty((n, n))
    for row, lat in enumerate(latvec):
        for col, lon in enumerate(lonvec):
            elevgrid[row, col] = reader.get_elevation(lat, lon)

    xgrid = (longrid - np.min(longrid)) * LONGITUDE_DEG_TO_METERS
    ygrid = (latgrid - np.min(latgrid)) * LATITUDE_DEG_TO_METERS
    zgrid = (elevgrid - np.min(elevgrid) + BOTTOM_SURFACE_OFFSET) * Z_SCALE

    # Define top vertices for each elevation point.
    top_vertices = (np.array((xgrid, ygrid, zgrid)).T).reshape(n * n, -1)

    # Define bottom vertices at 0 elevation.
    bottom_vertices = np.array([(x, y, 0) for x, y, _ in top_vertices])

    # Define total vertices.
    vertices = np.concatenate((top_vertices, bottom_vertices), 0)

    def get_vertices_index(lon_index, lat_index, is_top):
        return int((0 if is_top else n * n) + (lon_index * n + lat_index))

    # Define the triangle faces of the top surface.
    top_faces = np.empty(((n - 1) * (n - 1) * 2, 3), dtype=int)
    idx = 0
    for lon_index in range(n - 1):
        for lat_index in range(n - 1):
            top_faces[idx] = [
                get_vertices_index(lon_index, lat_index, True),
                get_vertices_index(lon_index + 1, lat_index, True),
                get_vertices_index(lon_index + 1, lat_index + 1, True),
            ]
            idx += 1
            top_faces[idx] = [
                get_vertices_index(lon_index, lat_index, True),
                get_vertices_index(lon_index + 1, lat_index + 1, True),
                get_vertices_index(lon_index, lat_index + 1, True),
            ]
            idx += 1

    # Define faces of bottom surface.
    bottom_faces = np.empty(((n - 1) * (n - 1) * 2, 3), dtype=int)
    idx = 0
    for lon_index in range(n - 1):
        for lat_index in range(n - 1):
            bottom_faces[idx] = [
                get_vertices_index(lon_index, lat_index, False),
                get_vertices_index(lon_index + 1, lat_index, False),
                get_vertices_index(lon_index + 1, lat_index + 1, False),
            ]
            idx += 1
            bottom_faces[idx] = [
                get_vertices_index(lon_index, lat_index, False),
                get_vertices_index(lon_index + 1, lat_index + 1, False),
                get_vertices_index(lon_index, lat_index + 1, False),
            ]
            idx += 1

    # Add faces for sides.
    side_faces = np.empty(((n - 1) * 2 * 2, 3), dtype=int)
    idx = 0
    for lon_index in range(n - 1):
        side_faces[idx] = [
            get_vertices_index(lon_index, 0, True),
            get_vertices_index(lon_index, 0, False),
            get_vertices_index(lon_index + 1, 0, False),
        ]
        idx += 1
        side_faces[idx] = [
            get_vertices_index(lon_index, 0, True),
            get_vertices_index(lon_index + 1, 0, False),
            get_vertices_index(lon_index + 1, 0, True),
        ]
        idx += 1
        side_faces[idx] = [
            get_vertices_index(lon_index, n - 1, True),
            get_vertices_index(lon_index, n - 1, False),
            get_vertices_index(lon_index + 1, n - 1, False),
        ]
        idx += 1
        side_faces[idx] = [
            get_vertices_index(lon_index, n - 1, True),
            get_vertices_index(lon_index + 1, n - 1, False),
            get_vertices_index(lon_index + 1, n - 1, True),
        ]
        idx += 1

    # Add faces for front/back.
    front_back_faces = np.empty(((n - 1) * 2 * 2, 3), dtype=int)
    idx = 0
    for lat_index in range(n - 1):
        front_back_faces[idx] = [
            get_vertices_index(0, lat_index, True),
            get_vertices_index(0, lat_index, False),
            get_vertices_index(0, lat_index + 1, False),
        ]
        idx += 1
        front_back_faces[idx] = [
            get_vertices_index(0, lat_index, True),
            get_vertices_index(0, lat_index + 1, False),
            get_vertices_index(0, lat_index + 1, True),
        ]
        idx += 1
        front_back_faces[idx] = [
            get_vertices_index(n - 1, lat_index, True),
            get_vertices_index(n - 1, lat_index, False),
            get_vertices_index(n - 1, lat_index + 1, False),
        ]
        idx += 1
        front_back_faces[idx] = [
            get_vertices_index(n - 1, lat_index, True),
            get_vertices_index(n - 1, lat_index + 1, False),
            get_vertices_index(n - 1, lat_index + 1, True),
        ]
        idx += 1

    # Create the mesh
    faces = np.concatenate(
        (top_faces, bottom_faces, side_faces, front_back_faces), 0)
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    mins = np.array([min([e[i] for e in vertices]) for i in range(3)])
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :] - mins

    # Write the mesh to file "cube.stl"
    cube.save(args.output_file)

    # Plot given range.
    # Create a new plot
    #    figure = plt.figure()
    #    axes = Axes3D(figure)
    #
    #    # Load the STL files and add the vectors to the plot
    #    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(cube.vectors))
    #
    #    # Auto scale to the mesh size
    #    axes.set_xlim3d(min([e[0] for v in cube.vectors for e in v]),
    #                    max([e[0] for v in cube.vectors for e in v]))
    #    axes.set_ylim3d(min([e[1] for v in cube.vectors for e in v]),
    #                    max([e[1] for v in cube.vectors for e in v]))
    #    axes.set_zlim3d(min([e[2] for v in cube.vectors for e in v]),
    #                    max([e[2] for v in cube.vectors for e in v]))

    # Show the plot to the screen
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(longrid,
                           latgrid,
                           elevgrid,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
