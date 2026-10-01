"""Geodetic utilities: WGS84 geodetic <-> ECEF <-> local ENU.

All functions are vectorised: inputs may be scalars or NumPy arrays and the
last axis of position arrays has length 3.

Frames
------
* Geodetic: latitude/longitude [deg], altitude above the WGS84 ellipsoid [m].
* ECEF: Earth-centred, Earth-fixed Cartesian [m].
* ENU: local tangent plane at a configurable origin, x=East, y=North, z=Up [m].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WGS84_A = 6378137.0                         # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563               # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)         # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)        # first eccentricity squared
WGS84_EP2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2  # second eccentricity squared
EARTH_MEAN_RADIUS = 6371008.8


def geodetic_to_ecef(lat_deg, lon_deg, alt_m) -> np.ndarray:
    """Convert geodetic coordinates to ECEF. Returns an array of shape (..., 3)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    alt = np.asarray(alt_m, dtype=np.float64)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    # Prime vertical radius of curvature N(phi).
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt) * cos_lat * np.cos(lon)
    y = (n + alt) * cos_lat * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.stack((x, y, z), axis=-1)


def ecef_to_geodetic(ecef) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ECEF to geodetic using Bowring's closed-form method.

    Bowring's parametric-latitude formula is accurate to well below a millimetre
    for points near the Earth's surface, which is all a UAV simulation needs.
    The ellipsoidal height uses the formulation that stays well-conditioned at
    the poles: ``h = p cos(phi) + (z + e^2 N sin(phi)) sin(phi) - N``.
    """
    ecef = np.asarray(ecef, dtype=np.float64)
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    p = np.hypot(x, y)
    lon = np.arctan2(y, x)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    lat = np.arctan2(z + WGS84_EP2 * WGS84_B * sin_t**3, p - WGS84_E2 * WGS84_A * cos_t**3)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    alt = p * cos_lat + (z + WGS84_E2 * n * sin_lat) * sin_lat - n
    return np.degrees(lat), np.degrees(lon), alt


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance on a spherical Earth [m] (for sanity checks / display)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    return 2.0 * EARTH_MEAN_RADIUS * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass(frozen=True)
class GeoOrigin:
    latitude: float
    longitude: float
    altitude: float


class GeoReference:
    """Local ENU tangent frame anchored at a geodetic origin.

    The rotation matrix rows are the East, North and Up unit vectors expressed
    in ECEF::

        R = [[-sin(lon0),            cos(lon0),           0        ],
             [-sin(lat0)cos(lon0), -sin(lat0)sin(lon0),  cos(lat0) ],
             [ cos(lat0)cos(lon0),  cos(lat0)sin(lon0),  sin(lat0) ]]

    so that ``enu = R @ (ecef - ecef0)`` and ``ecef = R.T @ enu + ecef0``.
    """

    def __init__(self, latitude: float, longitude: float, altitude: float = 0.0) -> None:
        self.origin = GeoOrigin(float(latitude), float(longitude), float(altitude))
        self._origin_ecef = geodetic_to_ecef(latitude, longitude, altitude)
        lat0, lon0 = np.radians(latitude), np.radians(longitude)
        sl, cl, so, co = np.sin(lat0), np.cos(lat0), np.sin(lon0), np.cos(lon0)
        self._rot = np.array(
            [
                [-so, co, 0.0],
                [-sl * co, -sl * so, cl],
                [cl * co, cl * so, sl],
            ]
        )

    def geodetic_to_enu(self, lat_deg, lon_deg, alt_m) -> np.ndarray:
        """Geodetic -> ENU. Returns (..., 3)."""
        delta = geodetic_to_ecef(lat_deg, lon_deg, alt_m) - self._origin_ecef
        return delta @ self._rot.T

    def enu_to_geodetic(self, enu) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ENU (..., 3) -> (lat_deg, lon_deg, alt_m)."""
        ecef = np.asarray(enu, dtype=np.float64) @ self._rot + self._origin_ecef
        return ecef_to_geodetic(ecef)

    def to_dict(self) -> dict[str, float]:
        return {"lat": self.origin.latitude, "lon": self.origin.longitude, "alt": self.origin.altitude}
