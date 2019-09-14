import numpy as np
from graphics import plot_3d
import grid

class DoaClasses():
    def __init__(self, doa_grid_resolution = np.pi/18):
        self.classes = self.generate_direction_classes(doa_grid_resolution)
        self.resolution = doa_grid_resolution
        self.resolution_deg = grid.rad2deg(doa_grid_resolution)
        self.all_pairs_deg = np.zeros((len(self.classes), 2))
        for i, angle in enumerate(self.classes):
            self.all_pairs_deg[i, :] = [angle.elevation_deg, angle.azimuth_deg]
    
    def xyz_at_index(self, index):
        assert index < len(self.classes)
        doa_class = self.classes[index]
        return doa_class.get_xyz_vector()

    def index_for_xyz(self, xyz):
        max_dot_product = -1
        class_index = -1
        for i, doa_class in enumerate(self.classes):
            dp = np.dot(xyz, doa_class.get_xyz_vector())
            if dp > max_dot_product:
                max_dot_product = dp
                class_index = i
        assert class_index > -1
        return class_index

    def generate_direction_classes(self, resolution_rad):
        direction_classes = []

        resolution_deg = grid.rad2deg(resolution_rad)
        elevation_deg, azimuth_deg = grid.makeDoaGrid(resolution_deg, fig=False)
        elevation_rad = grid.deg2rad(elevation_deg)
        azimuth_rad = grid.deg2rad(azimuth_deg)
        for el, az in zip(elevation_rad, azimuth_rad):
            direction_classes.append(DoaClass(el, az))

        return direction_classes

    def plot_classes(self):
        xs = []
        ys = []
        zs = []
        for doa_class in self.classes:
            xs.append(doa_class.x)
            ys.append(doa_class.y)
            zs.append(doa_class.z)
        plot_3d(xs, ys, zs)


class DoaClass():
    def __init__(self, elevation, azimuth):
        self.elevation = elevation
        self.azimuth = azimuth
        self.elevation_deg = grid.rad2deg(elevation)
        self.azimuth_deg = grid.rad2deg(azimuth)
        self.inclination = (np.pi/2) - self.elevation
        self.x = np.sin(self.inclination)*np.cos(self.azimuth)
        self.y = np.sin(self.inclination)*np.sin(self.azimuth)
        self.z = np.cos(self.inclination)

    def get_xyz_vector(self):
        return np.array([self.x, self.y, self.z])


def to_cartesian(x,doa_classes):
    assert x < len(doa_classes.classes)
    doa_class = doa_classes.classes[x]
    return doa_class.get_xyz_vector()


def to_class(xyz,doa_classes):
    max_dot_product = -1
    class_index = -1
    for i, doa_class in enumerate(doa_classes.classes):
        dp = np.dot(xyz, doa_class.get_xyz_vector())
        if dp > max_dot_product:
            max_dot_product = dp
            class_index = i
    assert class_index > -1
    return class_index


def lookup_class_index(elevation, azimuth, doa_classes):
    dist = grid.angularDistance(grid.rad2deg(elevation), grid.rad2deg(azimuth),
                                          doa_classes.all_pairs_deg[:, 0], doa_classes.all_pairs_deg[:, 1])
    class_index = np.argmin(dist)

    assert class_index > -1
    return class_index


def snap(x,doa_classes):
    return to_cartesian(to_class(x,doa_classes),doa_classes)


def snap_all(X,doa_classes):
    return [snap(x,doa_classes) for x in X]

