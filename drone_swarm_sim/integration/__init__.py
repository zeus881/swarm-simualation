"""Hardware adapter layer (Stage 5): real and SITL vehicles behind the platform's drone interface.

``hardware.vehicles`` in the configuration selects the link per vehicle:

* ``mavlink``: :class:`MAVLinkLink` over pymavlink (ArduPilot, PX4, any MAVLink autopilot / SITL),
* ``mavsdk``: :class:`MAVSDKLink` over the native MAVSDK binding (PX4 first; ArduPilot basics).

Both are wrapped by :class:`RemoteDrone`, a :class:`~simulation.drone.Drone` subclass, so swarm
behaviours, missions, alerts and the GCS drive simulated and real drones the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .remote import RemoteDrone, RemoteLink, RemoteState

if TYPE_CHECKING:
    from simulation.config import SimConfig
    from simulation.environment import Environment
    from simulation.events import EventBus
    from simulation.geo import GeoReference


def create_link(entry: dict[str, Any]) -> RemoteLink:
    """Link object for one ``hardware.vehicles`` entry (not started)."""
    if entry["type"] == "mavlink":
        from .mavlink_drone import MAVLinkLink
        return MAVLinkLink(entry["url"], target_system=entry.get("system_id"))
    from .mavsdk_drone import MAVSDKLink
    return MAVSDKLink(entry["url"])


def create_remote_drone(entry: dict[str, Any], drone_id: int, config: "SimConfig", environment: "Environment",
                        geo: "GeoReference", events: "EventBus", link: RemoteLink | None = None) -> RemoteDrone:
    link = link or create_link(entry)
    drone = RemoteDrone(drone_id, config, environment, geo, link=link, events=events, name=entry.get("name"))
    link.start()
    return drone


__all__ = ["RemoteDrone", "RemoteLink", "RemoteState", "create_link", "create_remote_drone"]
