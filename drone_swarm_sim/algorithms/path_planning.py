r"""Path planning around obstacles: A* on a 2-D occupancy grid, or RRT* in 3-D.

Both planners first test the straight segment; only a blocked segment triggers a search.

**A\*** (``obstacles.planner: astar``). The flight altitude is ``max(z_start, z_goal)``. A cell of a
grid around start and goal (``grid_resolution``, coarsened to stay under ``max_grid_cells``) is blocked
when an obstacle whose top reaches ``altitude - clearance`` lies within ``clearance + drone radius`` of
the cell centre (and, unless following terrain, when the terrain there is higher than
``altitude - terrain_clearance``). 8-connected A* with the octile heuristic
``h = res * (max(dx, dy) + (sqrt 2 - 1) * min(dx, dy))`` finds the shortest grid path. Line-of-sight
shortcutting (greedy: jump to the farthest waypoint with a clear segment) then turns it into a few
straight legs.

**RRT\*** (``obstacles.planner: rrtstar``). Samples 3-D points (10 % goal bias), extends by
``rrt_step``, keeps the parent that gives the lowest cost among nearby nodes
(``r = min(gamma (log n / n)^(1/3), 2.5 step)``), rewires neighbours through the new node, and keeps
improving the best goal connection until the iteration budget is used. Seeded, so plans are
deterministic. It can climb over low obstacles, which A* at a fixed altitude cannot.

Planning runs when a goto or mission leg starts, never every tick.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulation.config import ObstaclesConfig
    from simulation.environment import Environment

SQRT2 = math.sqrt(2.0)


@dataclass
class PlanResult:
    path: list[np.ndarray] | None          # waypoints after the start (last = goal); None = no path
    method: str                             # direct | astar | rrtstar
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> float:
        return 0.0 if not self.path else float(sum(np.linalg.norm(b - a) for a, b in zip(self.path[:-1], self.path[1:])))


class PathPlanner:
    def __init__(self, environment: "Environment", config: "ObstaclesConfig", drone_radius: float = 0.35) -> None:
        self.env = environment
        self.cfg = config
        self.radius = drone_radius

    # ------------------------------------------------------------------ checks
    def _terrain_ok(self, a: np.ndarray, b: np.ndarray, follow_terrain: bool) -> bool:
        terrain = self.env.terrain
        if terrain is None or follow_terrain:
            return True
        low = min(a[2], b[2]) - self.env.ground_level
        return terrain.max_along(a, b) + self.cfg.terrain_clearance <= low + 1e-6

    def segment_clear(self, a: np.ndarray, b: np.ndarray, clearance: float, follow_terrain: bool = False) -> bool:
        obstacles = self.env.obstacles
        if len(obstacles) and not obstacles.segment_clear(a, b, clearance):
            return False
        return self._terrain_ok(a, b, follow_terrain)

    # ------------------------------------------------------------------ entry point
    def plan(self, start: np.ndarray, goal: np.ndarray, *, clearance: float | None = None, method: str | None = None,
             follow_terrain: bool = False) -> PlanResult:
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        clearance = self.cfg.clearance if clearance is None else clearance
        if self.segment_clear(start, goal, clearance, follow_terrain):
            return PlanResult([goal], "direct")
        method = method or self.cfg.planner
        result = self._astar(start, goal, clearance, follow_terrain) if method == "astar" \
            else self._rrtstar(start, goal, clearance, follow_terrain)
        if result.ok:
            result.path = self._shortcut([start, *result.path], clearance, follow_terrain)[1:]
        return result

    def _shortcut(self, path: list[np.ndarray], clearance: float, follow_terrain: bool) -> list[np.ndarray]:
        """Greedy line-of-sight smoothing."""
        out, i = [path[0]], 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self.segment_clear(path[i], path[j], clearance, follow_terrain):
                j -= 1
            out.append(path[j])
            i = j
        return out

    # ------------------------------------------------------------------ A*
    def _astar(self, start: np.ndarray, goal: np.ndarray, clearance: float, follow_terrain: bool) -> PlanResult:
        env, cfg = self.env, self.cfg
        altitude = float(max(start[2], goal[2]))
        dist = float(np.hypot(*(goal[:2] - start[:2])))
        margin = max(120.0, 0.6 * dist)
        lo = np.maximum(np.minimum(start[:2], goal[:2]) - margin, env.bounds_min)
        hi = np.minimum(np.maximum(start[:2], goal[:2]) + margin, env.bounds_max)
        res = cfg.grid_resolution
        cells = ((hi - lo) / res).prod()
        if cells > cfg.max_grid_cells:
            res *= math.sqrt(cells / cfg.max_grid_cells)
        nx, ny = int(math.ceil((hi[0] - lo[0]) / res)) + 1, int(math.ceil((hi[1] - lo[1]) / res)) + 1
        xs = lo[0] + np.arange(nx) * res
        ys = lo[1] + np.arange(ny) * res
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        px, py = gx.ravel(), gy.ravel()
        blocked = np.zeros(nx * ny, dtype=bool)
        obs = env.obstacles
        inflate = clearance + self.radius
        for k, o in enumerate(obs.obstacles):
            if o.top + clearance <= altitude:                      # low enough to fly over
                continue
            if (o.x + o.bound_radius + inflate < lo[0] or o.x - o.bound_radius - inflate > hi[0]
                    or o.y + o.bound_radius + inflate < lo[1] or o.y - o.bound_radius - inflate > hi[1]):
                continue
            d_h, _ = obs._horizontal(px, py, np.full(len(px), k))
            blocked |= d_h < inflate
        if env.terrain is not None and not follow_terrain:
            blocked |= env.ground_heights(px, py) + cfg.terrain_clearance > altitude
        blocked = blocked.reshape(nx, ny)

        def cell(p: np.ndarray) -> tuple[int, int]:
            return (int(np.clip(round((p[0] - lo[0]) / res), 0, nx - 1)), int(np.clip(round((p[1] - lo[1]) / res), 0, ny - 1)))

        s, g = cell(start), cell(goal)
        s, g = self._free_near(blocked, s), self._free_near(blocked, g)
        if s is None or g is None:
            return PlanResult(None, "astar", "start or goal is enclosed by obstacles")
        # 8-connected A* on flat arrays
        gcost = np.full(nx * ny, np.inf)
        parent = np.full(nx * ny, -1, dtype=np.int64)
        sid, gid = s[0] * ny + s[1], g[0] * ny + g[1]
        gcost[sid] = 0.0
        heap = [(0.0, sid)]
        closed = np.zeros(nx * ny, dtype=bool)
        flat_blocked = blocked.ravel()
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
        gx_, gy_ = g
        while heap:
            _, cid = heapq.heappop(heap)
            if closed[cid]:
                continue
            if cid == gid:
                break
            closed[cid] = True
            cx, cy = divmod(cid, ny)
            base = gcost[cid]
            for dx, dy, w in moves:
                x2, y2 = cx + dx, cy + dy
                if x2 < 0 or y2 < 0 or x2 >= nx or y2 >= ny:
                    continue
                nid = x2 * ny + y2
                if closed[nid] or flat_blocked[nid]:
                    continue
                if dx and dy and (flat_blocked[cx * ny + y2] or flat_blocked[x2 * ny + cy]):
                    continue                                        # no corner cutting
                cost = base + w
                if cost < gcost[nid]:
                    gcost[nid] = cost
                    parent[nid] = cid
                    ddx, ddy = abs(x2 - gx_), abs(y2 - gy_)
                    heapq.heappush(heap, (cost + max(ddx, ddy) + (SQRT2 - 1) * min(ddx, ddy), nid))
        if not np.isfinite(gcost[gid]):
            return PlanResult(None, "astar", "no obstacle-free path found")
        cells_path = []
        cur = gid
        while cur != -1:
            cells_path.append(divmod(cur, ny))
            cur = parent[cur]
        cells_path.reverse()
        pts = [np.array([xs[i], ys[j], altitude]) for i, j in cells_path[1:-1]]
        return PlanResult([*pts, goal.copy()], "astar", f"{len(cells_path)} cells at {res:.1f} m")

    @staticmethod
    def _free_near(blocked: np.ndarray, c: tuple[int, int], max_ring: int = 12) -> tuple[int, int] | None:
        if not blocked[c]:
            return c
        nx, ny = blocked.shape
        for ring in range(1, max_ring + 1):
            best = None
            for i in range(c[0] - ring, c[0] + ring + 1):
                for j in (c[1] - ring, c[1] + ring) if abs(i - c[0]) != ring else range(c[1] - ring, c[1] + ring + 1):
                    if 0 <= i < nx and 0 <= j < ny and not blocked[i, j]:
                        d = (i - c[0]) ** 2 + (j - c[1]) ** 2
                        if best is None or d < best[0]:
                            best = (d, (i, j))
            if best:
                return best[1]
        return None

    # ------------------------------------------------------------------ RRT*
    def _rrtstar(self, start: np.ndarray, goal: np.ndarray, clearance: float, follow_terrain: bool) -> PlanResult:
        env, cfg = self.env, self.cfg
        rng = np.random.default_rng(cfg.seed)
        dist = float(np.linalg.norm(goal - start))
        margin = max(100.0, 0.5 * dist)
        lo = np.maximum(np.minimum(start[:2], goal[:2]) - margin, env.bounds_min)
        hi = np.minimum(np.maximum(start[:2], goal[:2]) + margin, env.bounds_max)
        z_lo = min(start[2], goal[2])
        z_hi = max(start[2], goal[2]) + 40.0
        step = cfg.rrt_step
        nodes = np.zeros((cfg.rrt_iterations + 2, 3))
        parent = np.full(len(nodes), -1, dtype=np.int64)
        cost = np.zeros(len(nodes))
        nodes[0] = start
        n = 1
        best_goal, best_cost = -1, math.inf
        gamma = 2.0 * max(dist, step) * 1.5
        for _ in range(cfg.rrt_iterations):
            sample = goal if rng.random() < 0.1 else np.array([rng.uniform(lo[0], hi[0]), rng.uniform(lo[1], hi[1]),
                                                              rng.uniform(z_lo, z_hi)])
            d = np.linalg.norm(nodes[:n] - sample, axis=1)
            near_i = int(np.argmin(d))
            direction = sample - nodes[near_i]
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            new = nodes[near_i] + direction * min(1.0, step / length)
            if not self.segment_clear(nodes[near_i], new, clearance, follow_terrain):
                continue
            radius = min(gamma * (math.log(n + 1) / (n + 1)) ** (1 / 3), 2.5 * step)
            dn = np.linalg.norm(nodes[:n] - new, axis=1)
            near = np.flatnonzero(dn <= radius)
            near = near[np.argsort(dn[near])][:10]
            best_p, best_c = near_i, cost[near_i] + float(np.linalg.norm(new - nodes[near_i]))
            for j in near:
                c = cost[j] + dn[j]
                if c < best_c and self.segment_clear(nodes[j], new, clearance, follow_terrain):
                    best_p, best_c = int(j), c
            nodes[n], parent[n], cost[n] = new, best_p, best_c
            for j in near:                                          # rewire through the new node
                c = best_c + dn[j]
                if c + 1e-9 < cost[j] and self.segment_clear(new, nodes[j], clearance, follow_terrain):
                    parent[j], cost[j] = n, c
            if np.linalg.norm(goal - new) <= step and best_c + np.linalg.norm(goal - new) < best_cost \
                    and self.segment_clear(new, goal, clearance, follow_terrain):
                best_goal, best_cost = n, best_c + float(np.linalg.norm(goal - new))
            n += 1
        if best_goal < 0:
            return PlanResult(None, "rrtstar", "no obstacle-free path found")
        chain = []
        cur = best_goal
        while cur > 0:
            chain.append(nodes[cur].copy())
            cur = parent[cur]
        chain.reverse()
        return PlanResult([*chain, goal.copy()], "rrtstar", f"{n} nodes, cost {best_cost:.0f} m")
