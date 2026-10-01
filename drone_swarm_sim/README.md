# Drone Swarm Simulation Platform

A modular multi-UAV swarm simulator for developing and testing swarm coordination, formation control,
collision avoidance, task allocation and autonomous missions. It has a real-time 3D Ground Control
Station in the browser.

The simulation core is pure Python/NumPy, deterministic and independent of the UI. Vehicles are
driven through a `DroneInterface` abstraction, so the same swarm logic flies simulated drones and
real / SITL vehicles (ArduPilot, PX4) over MAVLink or MAVSDK, in the same swarm.

> **Status: GCS upgrade stages 1–5 complete**: swarm intelligence, mission planning, operator
> awareness, realism (obstacles, terrain, radio link, sensors + Kalman filter, failure injection), and
> operations (mission replay, post-mission reports, hardware adapters, Operator/Observer login with an
> audit log). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

---

## Quick start

Requires Python 3.11+.

```bash
cd drone_swarm_sim
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python -m simulation --open       # starts the simulation + GCS and opens http://127.0.0.1:8000
```

Sign in as **operator / operator** (full control) or **observer / observer** (read-only). These are
demo accounts: change them before the GCS is reachable by anyone else (see *Stage 5* below).

Or with Docker (verified end to end: build, health check, login, commands, WebSocket telemetry, replay,
PDF report, audit log and recordings on the host volume):

```bash
docker compose up --build         # then open http://localhost:8000
```

The container listens on all interfaces, so change the demo passwords first. Recordings and
`audit.log` land in `./logs`, missions in `./data`, and `./configs` is mounted read-only.

The GCS needs no internet connection: Three.js is vendored under `frontend/vendor/`.

### Command-line options

```bash
python -m simulation --drones 25                          # swarm size
python -m simulation --set wind.speed=10 --set wind.direction=270
python -m simulation --set battery.drain_multiplier=20    # exercise low-battery failsafes quickly
python -m simulation --config my_scenario.yaml --port 9000
python -m simulation --headless --takeoff --duration 60   # no UI, prints a status line per second
python -m simulation.benchmark --drones 10 25 50 100      # stress test
```

## Using the Ground Control Station

| Action | How |
|---|---|
| Select a drone | click it in the 3D view or in the *Drone status* list |
| Multi-select | Ctrl+click; **All / None** buttons |
| Fly to a point | **Shift+click the ground**. Altitude and speed come from the *Mission* panel. A group keeps its shape. With a formation or flock active, Shift+click moves the whole group. |
| Formation | *Swarm* section: pick **Shape**, **Spacing**, **Heading** (empty = face travel), **Layers** / **Layer gap** and **Reference** (virtual point, or leader = first selected drone), then **Formation**. Change the shape at any time; drones re-slot on synchronized, non-crossing paths. |
| Custom shape | *Custom offsets*: type `forward, left, up` per line, or **From selection** to capture the selected drones' current layout, then **Fly custom**. |
| Flocking / release | **Flock** starts Reynolds flocking (tune the weights in *Flocking weights*); **Release** ends formation/flocking and members hold position. |
| Collision avoidance | toggle and **Method** (ORCA / potential field) in the *Swarm* section (ORCA on by default). The Mission panel shows deconflicted pairs, the hard floor with its breach count, the lowest separation of the run and the current leader. |
| Collapse / maximise | rail and dock chevrons; drag the dock's top edge; **M** maximises the 3D view. |
| Takeoff / Land / RTH / E-Stop | top bar (applies to the selection, or to **all** drones when nothing is selected), or the Inspector for one drone |
| Add / remove drones | **＋ Drone** / **－ Drone** (removes the selection, or the last drone) |
| Start / pause / reset | top bar, or **Space** to toggle |
| Shortcuts | `T` takeoff · `L` land · `H` return home · `X` emergency stop · `F` chase camera · `R` measure · `1`–`9` select group · `Esc` clear selection / stop measuring · `M` maximise view · `Space` start/pause (play/pause in REPLAY) · `←` / `→` replay ±5 s · `F1` / `F2` / `F3` FLY / PLAN / REPLAY |
| View | trails, flight paths, neighbour links, safety radius, labels, HUD, minimap, camera presets (Free / Top / Chase / Orbit) |

Drone colours: ONLINE (green), GROUNDED (blue), WARNING (amber), LOW BATTERY (orange),
COLLISION WARNING (magenta), COMM LOST (grey), EMERGENCY (flashing red).

## Stage 5: operations

* **Mission replay (REPLAY tab, `F3`).** Pick any recorded run in `logs/<run_id>` (the run being recorded
  now is listed too). Play, pause, ±10 s, a scrubber and speeds **0.25× – 8×**. Coloured **event markers**
  sit above the scrubber: red critical, amber warning, green commands, blue mission events. Hover a
  marker to read it, click it to jump there. The recorded frames drive the normal FLY view, so the 3D
  scene, fleet list, inspector, HUD, charts and minimap all work in replay. Live telemetry continues in
  the background and comes back when you leave the tab. Frames load in `replay.chunk_s` windows, so long
  50-drone runs play without loading the whole file into the browser.
* **Post-mission report.** One click: **Report** (Simulation section, current run) or **Report / PDF**
  in the replay bar. It is a self-contained HTML page or a PDF (A4) with:
  * the logo (`report.logo`; a GANDIV wordmark when the file is missing) and a summary
  * top-down flight paths with the geofence
  * battery curves with the thresholds, altitude, and minimum separation against the separation
    distance and the hard floor
  * separation violations, alerts, and the event timeline (showing who issued each command)
  * a per-drone table

  Both formats are generated offline with no extra packages: inline SVG, and a small built-in PDF writer.
* **Hardware adapter layer (`integration/`).** Add real or SITL vehicles in `hardware.vehicles`, one line
  each, choosing the adapter per vehicle:
  ```yaml
  hardware:
    vehicles:
      - {type: mavlink, url: "tcp:127.0.0.1:5760", name: SITL1}     # pymavlink: ArduPilot / PX4 / any MAVLink
      - {type: mavsdk,  url: "udpin://0.0.0.0:14540", name: PX4}    # native MAVSDK (mavsdk>=4)
  ```
  They join the swarm next to the simulated drones, tagged **MAV** / **SDK** in the 3D labels and the fleet
  list. Takeoff, land, RTL, goto (routes included), E-STOP, formations, flocking and collision avoidance
  all work: in a formation the pipeline velocity is streamed as setpoints at `hardware.setpoint_rate_hz`.
  A missing heartbeat for `hardware.heartbeat_timeout_s` makes the vehicle COMM LOST. Autopilot messages
  (pre-arm failures, rejected commands) appear in the event log. Tested against **3 ArduPilot SITL copters
  + 2 simulated drones** with both adapters:
  ```bash
  pip install pymavlink mavsdk
  python scripts/sitl_swarm.py --sitl-dir <ArduCopter SITL build> --instances 3 --link mavlink   # or mavsdk
  ```
* **Operator / Observer login.** With `security.enabled`, the GCS asks for a login.
  * **Operators** have full control. **Observers** see everything (live view, missions, replay,
    reports, audit), but every command control is inert and the server refuses their commands (REST
    403, WebSocket "read-only").
  * Passwords are stored as PBKDF2-SHA256 hashes in `security.users`. Make one with
    `python -m backend.security hash-password`.
  * Sessions are signed bearer tokens (`security.token_ttl_s`); repeated failed logins lock the account
    for `security.lockout_s`.
* **Command audit log.** Every command, simulation action, mission change, login, logout and denied
  request is written to `logs/audit.log`, one JSON line per entry: who (user, role, client), what (action,
  target drones, parameters, result) and when (UTC). The **Audit** dock tab shows it live. Command events
  in the event log and the reports also carry the user.
* **Confirmation dialogs.** **E-STOP** and **LAND** addressed to *all* drones (no selection; buttons or
  `X` / `L`), and **disabling the geofence** or removing its zones, ask for confirmation first.

## Stage 4: realism

All four models are on in the shipped `configs/simulation.yaml` and can be switched off one by one.

* **Obstacles.** Buildings, towers, trees and forests come from a YAML scene
  (`environment.scene_file`; an example ships as [configs/scenes/demo.yaml](configs/scenes/demo.yaml)).
  They are drawn in 3D and on the PLAN map. An obstacle-avoidance velocity stage keeps every drone
  `obstacles.clearance` away from surfaces, and a drone that touches one crashes, which is logged and
  alerted. Gotos, formation moves and mission legs are **routed around obstacles** with **A\*** (a 2-D
  grid at flight altitude) or **RRT\*** (3-D, can climb over), set by `obstacles.planner`.
* **Terrain.** Procedural hills, or a heightmap imported from **ESRI `.asc`**, **8/16-bit grayscale
  `.png`**, `.npy` or `.csv` (`terrain.source: file`). The ground is level around home. Shift+click
  gotos use the *Go-to alt* as **height above ground**. Missions can use **altitude mode AGL** (terrain
  following). The avoidance stage also keeps `obstacles.terrain_clearance` above the ground.
* **Communication.** The link to the GCS has a range limit, packet loss that rises to 100 % at the edge
  of range, latency and jitter. **COMM LOST** is real: no heartbeat for `timeout_s`, then the
  **failsafe** (RTL / LAND / HOLD) after `failsafe_timeout_s`. Commands travel the link with retries and
  come back as acknowledgements in the event log. The GCS shows the telemetry it actually received:
  delayed, and frozen with a growing age while the link is down (the label reads `NO LINK 7s`).
* **Sensors and Kalman filter.** GPS (noise plus a slowly wandering drift), barometer, compass and IMU
  feed a Kalman filter per drone, and **the autopilot flies on the estimate**. *View → Estimated position*
  shows the estimate as a ghost joined to the true position; the inspector shows the estimate error
  and 1σ.
* **Failure injection (instructor mode).** *Instructor* section: pick a failure (motor partial/total,
  GPS loss, comm loss, battery sag, wind gust) with its duration or strength, select drones, then
  **Inject failure**. Active failures are listed with a countdown and can be cleared one by one or all
  at once. Everything is logged and raises alerts.

## Stage 3: operator awareness (FLY view)

| Feature | Where / how |
|---|---|
| **HUD** | shown over the 3D view for the selected drone (toggle *View → HUD*): artificial horizon with pitch ladder and roll arc, heading tape, ground-speed and altitude tapes with a vertical-speed arrow, battery (%, voltage, predicted flight time left), flight mode / armed, GPS fix · satellites · HDOP and link-quality bars |
| **Telemetry charts** | dock tab **Charts**: altitude, speed, battery % and minimum separation over the last 120 s for the selected drones (up to 8 overlaid), or the swarm average when none is selected. The battery chart marks the RTL threshold; the separation chart marks the separation radius and the hard floor. Plain canvas, no chart library. |
| **Swarm health** | top of the right rail: counts of ONLINE / WARNING / LOW BATT / COMM LOST / FAILED, average and minimum battery, and the predicted flight time left of the airborne swarm (when the first drone will reach its emergency threshold at the power it is drawing now) |
| **Alerts** | dock tab **Alerts** with CRITICAL / WARNING / INFO priorities. A CRITICAL alert shows a flashing banner over the 3D view, beeps (repeats every 8 s until acknowledged; toggle *Audio*), and flashes the drones involved (red pulsing ring). **ACK** / **Ack all** acknowledge; repeats are de-duplicated (×count). Acknowledgement is a logged command and shared by every connected GCS. |
| **Minimap** | bottom-right of the 3D view: drones, home, geofence and the camera's position and view direction, auto-framed around the swarm. Click it to move the camera there. |
| **Measure** | ruler button in the view toolbar (or **R**): click two ground points for horizontal distance, bearing and ΔE / ΔN |
| **Camera presets** | *View* section: **Free**, **Top** (straight down over the swarm centre, wheel zooms), **Chase** (behind the selected drone; **F** toggles), **Orbit** (slow orbit of the swarm centre) |
| **Groups** | select drones, then **Save selection as group** (named Alpha, Bravo, …). Keys **1–9** select groups; chips show the key and size; × deletes. Groups are stored on the server and can be mission targets in the PLAN tab. |

Header tile *Min separation* now shows the true smallest distance between airborne drones (it is no
longer limited to the 10 m warning radius), and so does the inspector's *Nearest*.

## Stage 2: mission planning (PLAN tab, `missions/`)

Switch between the live 3D view and the planner with the **FLY / PLAN** tabs (or **F1 / F2**).

| Action | How (PLAN tab) |
|---|---|
| Add / move / delete waypoints | **Waypoint** tool: click the map. Drag a marker to move it. Right-click a marker to insert before or after, change its type, or delete it. Right-click a leg for *Insert waypoint here*. **Del** deletes the selected one. |
| Edit a waypoint | the table in the right rail: action (WAYPOINT, LOITER, TAKEOFF, LAND, RTL, CHANGE_FORMATION), altitude, speed and hold time. LOITER shows a radius and CHANGE_FORMATION a shape / spacing. |
| Pan / zoom | drag the map with **Select**, or middle-drag / Alt-drag with any tool; mouse wheel zooms; **Fit** frames the mission |
| Area survey | **Survey** tool: click the corners and click the first one (or double-click) to close. Set altitude, speed, number of drones, overlap % *or* line spacing, angle and finish, then **Generate**. The pattern is split into one strip per drone. |
| Geofence | **Fence** draws the inclusion polygon and **No-fly** adds exclusion zones. Drag corners to move them; right-click a corner to delete it. Set the breach action (RTL / LAND / HOLD) and ceiling, then **Apply fence**. The fence shows in 3D as translucent walls. |
| Fly it | choose **Assign to** (selected drones, whole swarm or a group) and **Fly in formation**, then **Upload & Fly**. The mission is validated first; warnings (fence, altitude, battery) open a confirmation dialog. **Pause / Resume / Abort** control the run. Progress shows in both tabs. |
| Files | **Save** / **Load** use the schema-validated library (`data/missions/mission_*.json`). **Download** saves JSON locally. **Export QGC** writes QGC WPL 110 `.waypoints` (one file per track) for Mission Planner / QGroundControl. **Import file…** opens `.waypoints` or `.json`. |
| Measure | **Measure** tool: click two points for distance and bearing |

Behaviour notes:
* The swarm flies a single-track mission **in its current formation** (it forms the configured default
  if none is active). A split survey flies one track per drone.
* Drones on the ground take off automatically. An operator command, a failsafe or a geofence breach
  takes a drone out of its mission; the rest carry on.
* The geofence is **predictive**: it checks where each drone will be in 2 s, so HOLD stops a drone before
  it enters a no-fly zone.
* Pre-flight validation estimates distance, time and energy with the battery model. In tests this came
  within 25 % of the simulated energy.

## Stage 1: swarm intelligence (`algorithms/`)

* **Collision avoidance** (`collision_avoidance.py`, `orca.py`) runs for every airborne drone in every mode,
  as the highest-priority velocity stage, in three layers:
  1. **ORCA** velocity obstacles (primary, `swarm.avoidance_method: orca`): reciprocal half-spaces over a
     3 s horizon, solved with the RVO2-3D linear program.
  2. **Potential field** with closest-point-of-approach prediction and yielding. It is the fallback when a
     drone's ORCA program is infeasible, and can also be selected as the method.
  3. A **hard safety filter** on the braking distance, which guarantees that no two drones get closer than
     `swarm.min_separation` (default 4 m). The tests prove it with ORCA *and* the potential field switched
     off: drones rammed together at 15 m/s each, and a 16-drone swarm imploding onto its centroid, all stop
     at 4.3 m.
* **Formations** (`formation.py`): line, column, V, diamond, grid, circle, wedge and custom, with
  **spacing**, fixed or travel-aligned **heading**, and **altitude layers**. Custom shapes come from the
  config, from typed offsets, or from the positions of the selected drones ("From selection").
  Drones are matched to slots with the **Hungarian algorithm** on squared distance. Shape changes are
  **synchronized straight-line transitions** (CAPT): every drone starts where it is and all arrive together,
  so paths never cross. A **virtual reference** moves with `swarm_goto` and slows while members lag behind.
* **Leader-follower**: the formation tracks a leader you fly normally. If the leader fails, loses its link,
  lands or returns home, the follower closest to the lead slot is **promoted automatically**. It inherits
  the leader's goto, and a `leader_promoted` event is logged.
* **Reynolds flocking** (`flocking.py`): `V = W1·separation + W2·alignment + W3·cohesion + W4·goal (+ W5·obstacle)`,
  with the weights tunable at runtime (GCS sliders or `set_flocking_weights`).
* **Operator authority**: any command or failsafe (e.g. battery RTL) takes a drone out of the formation;
  the remaining members re-slot automatically. Switching avoidance off is an explicit operator override.

"Every drone flies through one point" stress scenario (40 s, `min_separation` 4 m, separation radius 5 m):

| | 10 drones | 25 drones | 50 drones |
|---|---|---|---|
| ORCA (default): lowest separation / safety-radius entries / floor breaches | 6.9 m / 0 / 0 | 6.8 m / 0 / 0 | 5.0 m / 1 / 0 |
| Potential field: lowest separation / entries / floor breaches | 5.7 m / 0 / 0 | 5.1 m / 0 / 0 | 4.6 m / 4 / 0 |

**Exit test:** 25 drones switch through all 8 formations (V → grid → circle → line → diamond → wedge → column
→ custom → V) with **zero** separation violations. No pair even comes inside the 10 m warning radius
(`tests/test_algorithms.py::test_shape_changes_with_25_drones_never_violate_separation`).

## Phase 1 features

* **Vehicle model**: point-mass quadrotor with tilt- and thrust-limited, first-order-lagged thrust,
  drag relative to the air, ground contact and hard-landing detection. A PI velocity controller with
  anti-windup holds position against wind. Guidance uses a minimum-time braking profile. Drones never
  teleport.
* **Autopilot modes**: DISARMED, ARMED, TAKEOFF, HOVER, GOTO, OFFBOARD (velocity setpoints with a
  timeout failsafe), LAND, RTL (climb → return → land), EMERGENCY (descend, or motor kill if configured).
  Also auto-disarm and a geofence.
* **Battery**: energy model (hover ∝ mass^1.5, air-relative speed, climb, manoeuvring, payload, wind),
  voltage sag, and failsafes at 30 % (warning), 20 % (return home) and 10 % (emergency land).
* **Environment**: world bounds and ceiling, home base with per-drone landing pads, wind with altitude
  shear and Ornstein-Uhlenbeck gusts.
* **Swarm manager**: dynamic add/remove, KD-tree neighbour search, collision detection
  (WARNING < 10 m, AVOIDANCE < 5 m, COLLISION < 1 m), and a prioritised velocity pipeline where
  Phase 2 plugs in its algorithms.
* **Coordinates**: WGS84 ⇄ ECEF ⇄ ENU conversion with a configurable origin. Every drone reports
  local XYZ and lat/lon/alt, and `goto` accepts either.
* **Backend**: FastAPI REST API (`/docs`) and a WebSocket telemetry stream at 20 Hz with a
  command channel.
* **Recording**: `logs/simulation.log`, `logs/audit.log`, plus one folder per run with `telemetry.csv`,
  `mission_events.json`, `collision_events.json`, `config.json` and `world.json`. Everything is streamed
  to disk, so memory stays flat on long runs. The event files are JSON arrays that stay readable while
  they are being written.

## Project layout

```text
backend/        FastAPI app: api/ (REST incl. auth, replay), websocket/ (telemetry hub), security.py (login, audit)
simulation/     engine, runner, drone, dynamics, guidance, battery, environment, swarm, spatial, geo,
                recorder, replay, report, ...
integration/    hardware adapters: remote.py (RemoteDrone), mavlink_drone.py, mavsdk_drone.py
scripts/        sitl_swarm.py (ArduPilot SITL end-to-end check)
frontend/       index.html, src/*.js (ES modules, no build step), styles/, vendor/three
configs/        simulation.yaml, documented default configuration
docs/           ARCHITECTURE.md, design, math models, API, phases
tests/          pytest suite
logs/           runtime output
```

`algorithms/` holds `formation.py`, `flocking.py`, `orca.py`, `collision_avoidance.py`,
`obstacle_avoidance.py`, `path_planning.py` and `coordinator.py` (swarm modes and their commands).
`missions/` holds the mission model, manager, survey generator, geofence, validation, QGC WPL 110
support and the file library. `estimation/` holds the batched Kalman filter and the navigation filter.
The realism models (`terrain.py`, `obstacles.py`, `communication.py`, `sensors.py`, `failures.py`) live
in `simulation/`.

## Configuration

Every parameter lives in [configs/simulation.yaml](configs/simulation.yaml). Precedence:
built-in defaults → YAML file (`--config` or `$SWARM_CONFIG`) → `--set section.key=value`.
Unknown keys are rejected so typos fail loudly, and ranges plus cross-field rules are validated at startup.

## API

REST (interactive docs at `http://127.0.0.1:8000/docs`). With `security.enabled`, log in first and send
the token as `Authorization: Bearer <token>` (or `?token=` on download links). Reads need any role;
anything that changes state needs an operator:

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/auth/login -H "Content-Type: application/json" \
        -d '{"username": "operator", "password": "operator"}' | python -c "import sys,json; print(json.load(sys.stdin)['token'])")
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/replay/runs            # recorded runs
curl -H "Authorization: Bearer $TOKEN" -o report.pdf \
     "http://127.0.0.1:8000/api/replay/runs/<run_id>/report?format=pdf"                  # post-mission report
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/audit?limit=20          # audit log
```

The examples below omit the header (as with `security.enabled: false`):

```bash
curl http://127.0.0.1:8000/api/status
curl http://127.0.0.1:8000/api/drones/3
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" \
     -d '{"type": "takeoff", "drone_ids": null, "params": {"altitude": 25}}'
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" \
     -d '{"type": "goto", "drone_ids": [1, 2], "params": {"position": [150, 80, 30], "speed": 8}}'
curl -X POST http://127.0.0.1:8000/api/simulation/pause
```

Commands: `arm, disarm, takeoff, land, hover, goto, set_velocity, set_heading, set_altitude,
return_to_home, emergency_stop, add_drone, remove_drone, set_wind`; for swarm behaviour
`set_formation, swarm_goto, start_flocking, release_swarm, set_avoidance, set_flocking_weights`;
for missions `mission_validate, mission_start, mission_pause, mission_resume, mission_abort,
set_geofence, clear_geofence, define_group, delete_group` (`GET /api/commands` lists them with their
parameters). Mission files, surveys and QGC import/export live under `/api/missions/*`
(see [ARCHITECTURE.md §7.5](docs/ARCHITECTURE.md)).

```bash
# fly a small patrol with the whole swarm in formation
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" -d '{"type": "mission_start",
  "params": {"mission": {"name": "patrol", "waypoints": [{"action": "TAKEOFF", "alt": 25},
  {"x": 150, "y": 0, "alt": 30, "hold": 5}, {"x": 150, "y": 150, "alt": 30}, {"action": "RTL"}]}}}'
# a no-fly zone that makes drones hold before entering
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" -d '{"type": "set_geofence",
  "params": {"enabled": true, "action": "HOLD", "exclusions": [{"name": "Tower", "polygon": [[100,-50],[150,-50],[150,50],[100,50]]}]}}'
```

```bash
# V formation for all airborne drones, 20 m spacing, two altitude layers, then move it
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" \
     -d '{"type": "set_formation", "params": {"shape": "v", "spacing": 20, "layers": 2, "layer_spacing": 8}}'
# custom shape from offsets [forward, left, up]
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" \
     -d '{"type": "set_formation", "params": {"shape": "custom", "offsets": [[0,0,0],[-12,12,0],[-12,-12,0]]}}'
# switch avoidance method
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" \
     -d '{"type": "set_avoidance", "params": {"method": "potential_field"}}'
curl -X POST http://127.0.0.1:8000/api/commands -H "Content-Type: application/json" \
     -d '{"type": "swarm_goto", "params": {"position": [300, 150, 40]}}'
```

Each telemetry frame also carries `swarm_control`: the mode, avoidance state, formation (shape, slots,
reference, heading, maximum slot error) and flocking goal and weights.

The WebSocket protocol (`ws://host:8000/ws/telemetry`) is documented in
[docs/ARCHITECTURE.md §7.2](docs/ARCHITECTURE.md). A per-drone telemetry record:

```json
{"drone_id": 1, "name": "D01",
 "position": {"x": 125.4, "y": 82.1, "z": 40.2}, "gps": {"lat": 47.3988, "lon": 8.5473, "alt": 528.2},
 "velocity": {"x": 4.2, "y": 1.4, "z": 0.1}, "heading": 135.2, "roll": -2.1, "pitch": -6.3,
 "battery": 87.4, "battery_state": "NORMAL", "mode": "GOTO", "armed": true, "health": "OK",
 "task": "GOTO", "communication": "ONLINE", "target": {"x": 200, "y": 90, "z": 40},
 "neighbors": [2, 4], "collision_state": "CLEAR", "...": "..."}
```

## Using the engine from Python

```python
from simulation import SimulationEngine, load_config

engine = SimulationEngine(load_config(overrides={"simulation.drone_count": 5}), record=False)
engine.execute({"type": "takeoff", "params": {"altitude": 20}})
engine.run_for(10.0)                               # as fast as possible, deterministic
engine.execute({"type": "goto", "params": {"position": [100, 50, 30]}})
engine.run_for(20.0)
print(engine.swarm.summary())
```

Custom swarm behaviour plugs in as a velocity-pipeline stage, without touching the drone class:

```python
from simulation.swarm import VelocityStage

class Hold10mAboveGround(VelocityStage):
    name, priority = "demo", 20
    def apply(self, ctx, v):
        out = v.copy()
        low = ctx.controllable & (ctx.positions[:, 2] < 10)
        out[low, 2] = 1.0
        return out

engine.swarm.add_stage(Hold10mAboveGround())
```

## Tests and benchmarks

```bash
python -m pytest                     # full suite (323 tests; the ArduPilot SITL test runs only with $SWARM_SITL_DIR)
python -m pytest -m "not slow"       # skip real-time budget tests
python -m simulation.benchmark       # 10 / 25 / 50 / 100 drones -> logs/benchmark.json
```

Measured on the development machine (Windows 11, i7-1185G7 on mains power, Python 3.14, 30 Hz,
all drones flying at once, collision avoidance on):

| Drones | Step mean | Step p95 | Max sustainable rate | Snapshot size |
|---:|---:|---:|---:|---:|
| 10 | 1.5 ms | 1.5 ms | ~690 Hz | 12 kB |
| 25 | 3.4 ms | 3.6 ms | ~290 Hz | 22 kB |
| 50 | 7.3 ms | 10.5 ms | ~140 Hz | 39 kB |
| 100 | 15.0 ms | 19.4 ms | ~67 Hz | 76 kB |

All sizes fit inside the 33 ms budget at 30 Hz. With every Stage 4 realism model on (Stage 5 run, mains
power), 50 drones measure 11.6 ms mean / 23.3 ms p95. A 30-minute real-time soak with 50 drones, recording
and a WebSocket client held 30 Hz, and RSS stayed flat (peak 115 MB, no upward trend). On battery power Windows throttles this CPU to about
60 %, and the 25- and 50-drone real-time tests can then fail; run them on mains power.

## Roadmap

| Phase | Scope |
|---|---|
| 1 ✅ | Core engine, drone model, battery, wind, swarm registry, collision detection, backend, 3D GCS |
| 2 ✅ | Formations (line, column, V, diamond, grid, circle, wedge, custom), leader-follower, flocking, collision avoidance, GCS swarm controls |
| 3 | Obstacles, terrain, no-fly zones, dynamic obstacles, path planning |
| 4 | Sensors, Kalman filter, communication model (loss, latency, range), failure injection |
| 5 | Targets and tracking, task allocation (auction), mission manager, example mission, search |
| 6 | Replay, extended stress tests, PX4 SITL / MAVLink adapters, Docker hardening (done in GCS stage 5) |

### Dependencies added per stage

| Stage | New dependencies |
|---|---|
| 1–4 | none (NumPy / SciPy / FastAPI already required) |
| 5 | optional `pymavlink>=2.4.40` and `mavsdk>=4.0` for the hardware adapters (`pip install .[hardware]`; also in `requirements.txt` and the Docker image). Replay, reports, login and audit use the standard library only. |
