"""Terrain: a height grid stretched over the world bounds, with bilinear interpolation.

Sources (``terrain.source``):

* ``procedural``: smooth random hills (sum of Gaussians, seeded, deterministic), flattened around the
  home base so the landing pads are level.
* ``file``: a heightmap file, stretched over the world bounds (row 0 = north edge, as in images and
  DEMs):

  - ``.asc``  ESRI ASCII grid (the usual DEM exchange format; ``NODATA`` cells become the minimum),
  - ``.npy``  NumPy array of heights in metres,
  - ``.csv`` / ``.txt``  comma / whitespace separated heights in metres,
  - ``.png``  8- or 16-bit grayscale PNG, scaled so the brightest pixel is ``max_height`` (decoded
    with the standard library only; no imaging dependency).

``height(x, y) = offset + vertical_scale * bilinear(grid, x, y)``; outside the grid the edge value is
used. All heights are metres in the local ENU frame, added to ``environment.ground_level``.
"""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .config import TerrainConfig


class TerrainError(ValueError):
    """Unreadable or invalid heightmap."""


# ----------------------------------------------------------------------------- loaders

def _read_png_gray(path: Path) -> np.ndarray:
    """Minimal PNG decoder: non-interlaced 8/16-bit grayscale (or RGB/RGBA, averaged), all filter types."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise TerrainError(f"{path.name} is not a PNG file")
    pos, idat, header = 8, bytearray(), None
    while pos < len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            header = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            idat.extend(chunk)
        elif ctype == b"IEND":
            break
    if header is None:
        raise TerrainError(f"{path.name}: missing IHDR")
    width, height, depth, color, _, _, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if channels is None or depth not in (8, 16) or interlace:
        raise TerrainError(f"{path.name}: only non-interlaced 8/16-bit grayscale/RGB(A) PNGs are supported")
    bpp = channels * depth // 8
    stride = width * bpp
    raw = zlib.decompress(bytes(idat))
    out = np.zeros((height, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for r in range(height):
        f = raw[r * (stride + 1)]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=r * (stride + 1) + 1).astype(np.int32)
        if f == 0:
            cur = line
        elif f == 2:
            cur = (line + prev) & 0xFF
        else:                                              # filters that depend on the left neighbour
            cur = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                left = cur[i - bpp] if i >= bpp else 0
                up = prev[i]
                if f == 1:
                    pred = left
                elif f == 3:
                    pred = (left + up) >> 1
                elif f == 4:
                    ul = prev[i - bpp] if i >= bpp else 0
                    p = left + up - ul
                    pa, pb, pc = abs(p - left), abs(p - up), abs(p - ul)
                    pred = left if pa <= pb and pa <= pc else (up if pb <= pc else ul)
                else:
                    raise TerrainError(f"{path.name}: bad PNG filter {f}")
                cur[i] = (line[i] + pred) & 0xFF
        out[r] = cur
        prev = cur
    if depth == 16:
        pix = out.reshape(height, width * channels, 2)
        values = (pix[..., 0].astype(np.float64) * 256 + pix[..., 1]).reshape(height, width, channels)
        full = 65535.0
    else:
        values = out.reshape(height, width, channels).astype(np.float64)
        full = 255.0
    gray = values[..., :min(channels, 3)].mean(axis=2) if channels >= 3 else values[..., 0]
    return gray / full


def _read_asc(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    meta: dict[str, float] = {}
    k = 0
    while k < len(lines) and lines[k].split() and lines[k].split()[0].replace("_", "").isalpha():
        key, value = lines[k].split()[:2]
        meta[key.lower()] = float(value)
        k += 1
    try:
        rows = [[float(v) for v in ln.split()] for ln in lines[k:] if ln.strip()]
        grid = np.array(rows, dtype=np.float64)
    except ValueError as exc:
        raise TerrainError(f"{path.name}: bad number in the grid ({exc})") from None
    nodata = meta.get("nodata_value")
    if nodata is not None:
        mask = grid == nodata
        if mask.all():
            raise TerrainError(f"{path.name}: grid contains only NODATA")
        grid[mask] = grid[~mask].min()
    return grid


def load_heightmap(path: Path, max_height: float) -> np.ndarray:
    """Heights in metres, row 0 = north edge."""
    if not path.is_file():
        raise TerrainError(f"heightmap not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".png":
        grid = _read_png_gray(path) * max_height
    elif suffix == ".asc":
        grid = _read_asc(path)
    elif suffix == ".npy":
        grid = np.load(path, allow_pickle=False).astype(np.float64)
    elif suffix in (".csv", ".txt"):
        text = path.read_text(encoding="utf-8").replace(",", " ")
        try:
            grid = np.array([[float(v) for v in ln.split()] for ln in text.splitlines() if ln.strip()], dtype=np.float64)
        except ValueError as exc:
            raise TerrainError(f"{path.name}: bad number ({exc})") from None
    else:
        raise TerrainError(f"unsupported heightmap format '{suffix}' (use .asc, .png, .npy or .csv)")
    if grid.ndim != 2 or min(grid.shape) < 2:
        raise TerrainError(f"{path.name}: heightmap must be a 2-D grid of at least 2 x 2")
    if not np.all(np.isfinite(grid)):
        raise TerrainError(f"{path.name}: heightmap contains non-finite values")
    return grid


def procedural_heights(size_x: float, size_y: float, resolution: float, seed: int, hills: int, hill_radius: float,
                       max_height: float) -> np.ndarray:
    """Smooth random hills (row 0 = north edge)."""
    rng = np.random.default_rng(seed)
    cols = int(round(size_x / resolution)) + 1
    rows = int(round(size_y / resolution)) + 1
    xs = np.linspace(-size_x / 2, size_x / 2, cols)
    ys = np.linspace(size_y / 2, -size_y / 2, rows)
    gx, gy = np.meshgrid(xs, ys)
    h = np.zeros_like(gx)
    for _ in range(hills):
        cx, cy = rng.uniform(-size_x / 2, size_x / 2), rng.uniform(-size_y / 2, size_y / 2)
        r = hill_radius * rng.uniform(0.6, 1.6)
        amp = max_height * rng.uniform(0.35, 1.0)
        h += amp * np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * r * r))
    return np.minimum(h, max_height)


# ----------------------------------------------------------------------------- terrain

class Terrain:
    """Height field over the world bounds. ``grid`` rows run south -> north after loading."""

    def __init__(self, grid_north_up: np.ndarray, x_min: float, y_min: float, size_x: float, size_y: float,
                 *, vertical_scale: float = 1.0, offset: float = 0.0) -> None:
        g = np.asarray(grid_north_up, dtype=np.float64)[::-1] * vertical_scale + offset   # row 0 = south
        self.grid = np.ascontiguousarray(g)
        self.rows, self.cols = self.grid.shape
        self.x_min, self.y_min = float(x_min), float(y_min)
        self.dx = size_x / (self.cols - 1)
        self.dy = size_y / (self.rows - 1)
        self.min_height = float(self.grid.min())
        self.max_height = float(self.grid.max())
        self._g = self.grid                                      # local alias for the scalar fast path

    @classmethod
    def from_config(cls, cfg: "TerrainConfig", size_x: float, size_y: float, home_xy: tuple[float, float],
                    project_root: Path) -> "Terrain | None":
        if not cfg.enabled:
            return None
        if cfg.source == "procedural":
            grid = procedural_heights(size_x, size_y, cfg.resolution, cfg.seed, cfg.hills, cfg.hill_radius, cfg.max_height)
        else:
            path = Path(cfg.file)
            grid = load_heightmap(path if path.is_absolute() else project_root / path, cfg.max_height)
        terrain = cls(grid, -size_x / 2, -size_y / 2, size_x, size_y, vertical_scale=cfg.vertical_scale, offset=cfg.offset)
        if cfg.flatten_radius > 0:
            terrain.flatten(home_xy[0], home_xy[1], cfg.flatten_radius)
        return terrain

    def flatten(self, x: float, y: float, radius: float) -> None:
        """Level a disc around (x, y) to its centre height with a smooth rim (keeps landing pads level)."""
        ys = self.y_min + np.arange(self.rows) * self.dy
        xs = self.x_min + np.arange(self.cols) * self.dx
        gx, gy = np.meshgrid(xs, ys)
        r = np.hypot(gx - x, gy - y)
        centre = self.height(x, y)
        w = np.clip((r - radius) / radius, 0.0, 1.0)            # 0 inside the disc, 1 beyond 2 radii
        w = w * w * (3 - 2 * w)
        self.grid = np.ascontiguousarray(centre * (1 - w) + self.grid * w)
        self._g = self.grid
        self.min_height = float(self.grid.min())
        self.max_height = float(self.grid.max())

    def height(self, x: float, y: float) -> float:
        """Bilinear height at one point (scalar fast path used every physics step)."""
        fx = (x - self.x_min) / self.dx
        fy = (y - self.y_min) / self.dy
        fx = 0.0 if fx < 0 else (self.cols - 1.0 if fx > self.cols - 1 else fx)
        fy = 0.0 if fy < 0 else (self.rows - 1.0 if fy > self.rows - 1 else fy)
        c, r = min(int(fx), self.cols - 2), min(int(fy), self.rows - 2)   # last cell: t = 1 (exact at the edge)
        tx, ty = fx - c, fy - r
        g = self._g
        h00, h01 = g[r, c], g[r, c + 1]
        h10, h11 = g[r + 1, c], g[r + 1, c + 1]
        return float((h00 * (1 - tx) + h01 * tx) * (1 - ty) + (h10 * (1 - tx) + h11 * tx) * ty)

    def heights(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised bilinear heights."""
        fx = np.clip((np.asarray(xs, dtype=np.float64) - self.x_min) / self.dx, 0, self.cols - 1)
        fy = np.clip((np.asarray(ys, dtype=np.float64) - self.y_min) / self.dy, 0, self.rows - 1)
        c, r = np.minimum(fx.astype(int), self.cols - 2), np.minimum(fy.astype(int), self.rows - 2)
        tx, ty = fx - c, fy - r
        g = self.grid
        return (g[r, c] * (1 - tx) + g[r, c + 1] * tx) * (1 - ty) + (g[r + 1, c] * (1 - tx) + g[r + 1, c + 1] * tx) * ty

    def max_along(self, a: np.ndarray, b: np.ndarray, step: float | None = None) -> float:
        """Highest terrain point on the horizontal segment a -> b."""
        step = step or min(self.dx, self.dy) / 2
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1)
        t = np.linspace(0.0, 1.0, n)
        return float(self.heights(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t).max())

    def sample_grid(self, max_n: int = 129) -> dict[str, Any]:
        """Down-sampled grid (row 0 = south) for clients to render."""
        rs = np.linspace(0, self.rows - 1, min(self.rows, max_n)).round().astype(int)
        cs = np.linspace(0, self.cols - 1, min(self.cols, max_n)).round().astype(int)
        sub = self.grid[np.ix_(rs, cs)]
        return {"rows": len(rs), "cols": len(cs), "x_min": self.x_min, "y_min": self.y_min,
                "size_x": self.dx * (self.cols - 1), "size_y": self.dy * (self.rows - 1),
                "heights": np.round(sub, 2).ravel().tolist(), "min": self.min_height, "max": self.max_height}

    def describe(self) -> dict[str, Any]:
        return {"enabled": True, "rows": self.rows, "cols": self.cols, "min": round(self.min_height, 2),
                "max": round(self.max_height, 2), "cell": [round(self.dx, 3), round(self.dy, 3)]}
