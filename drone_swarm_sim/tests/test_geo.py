import numpy as np
import pytest

from simulation.geo import GeoReference, ecef_to_geodetic, geodetic_to_ecef, haversine_distance


def test_ecef_known_point():
    # Equator / prime meridian at sea level lies on the X axis at the semi-major axis.
    np.testing.assert_allclose(geodetic_to_ecef(0.0, 0.0, 0.0), [6378137.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.parametrize("lat,lon,alt", [(47.397742, 8.545594, 488.0), (-33.9, 151.2, 30.0), (89.9, -120.0, 1000.0),
                                         (0.0, 179.99, -50.0)])
def test_geodetic_ecef_roundtrip_submillimetre(lat, lon, alt):
    la, lo, al = ecef_to_geodetic(geodetic_to_ecef(lat, lon, alt))
    assert abs(la - lat) < 1e-9 and abs(lo - lon) < 1e-9 and abs(al - alt) < 1e-3


def test_enu_origin_is_zero_and_axes_point_east_north_up():
    geo = GeoReference(47.4, 8.5, 400.0)
    np.testing.assert_allclose(geo.geodetic_to_enu(47.4, 8.5, 400.0), [0, 0, 0], atol=1e-6)
    north = geo.geodetic_to_enu(47.401, 8.5, 400.0)
    east = geo.geodetic_to_enu(47.4, 8.501, 400.0)
    up = geo.geodetic_to_enu(47.4, 8.5, 500.0)
    assert north[1] > 100 and abs(north[0]) < 1e-6
    assert east[0] > 70 and abs(east[1]) < 0.01
    np.testing.assert_allclose(up, [0, 0, 100], atol=1e-6)


def test_enu_distance_matches_haversine():
    geo = GeoReference(47.4, 8.5, 0.0)
    p = geo.geodetic_to_enu(47.41, 8.52, 0.0)
    d = haversine_distance(47.4, 8.5, 47.41, 8.52)
    assert abs(np.hypot(p[0], p[1]) - d) / d < 5e-3


def test_enu_roundtrip_vectorised():
    geo = GeoReference(47.397742, 8.545594, 488.0)
    pts = np.array([[0, 0, 0], [1000, -500, 50], [-900, 900, 300]], dtype=float)
    lat, lon, alt = geo.enu_to_geodetic(pts)
    back = geo.geodetic_to_enu(lat, lon, alt)
    np.testing.assert_allclose(back, pts, atol=1e-3)
