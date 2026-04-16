"""Shadow-mode PCLA agent runner.

Attaches multiple PCLA agents to the same follower vehicle at once. Each
agent brings its own sensor suite (attached by PCLA internally) and is
invoked every simulation tick via `get_action()`. The returned
`carla.VehicleControl` is logged to CSV but NOT applied to the vehicle —
the vehicle is driven by the scripted cruise controller in the scenario.

Per-agent CSVs end up next to the run's `telemetry.csv`:
    agent_<name>.csv   columns: tick, sim_time_s, throttle, steer, brake
"""

import csv
import os
import sys
import traceback

import carla

from common import PCLA_DIR

if PCLA_DIR not in sys.path:
    sys.path.insert(0, PCLA_DIR)

from PCLA import PCLA  # noqa: E402
from leaderboard_codes.carla_data_provider import CarlaDataProvider  # noqa: E402


# PCLA registers the ego vehicle inside setup_sensors(); with N agents on
# the same vehicle, the 2nd+ would raise KeyError. Idempotent patch = single
# shared registration across all shadow agents.
_original_register_actor = CarlaDataProvider.register_actor


def _idempotent_register_actor(actor):
    if actor in CarlaDataProvider._actor_velocity_map:
        return
    _original_register_actor(actor)


CarlaDataProvider.register_actor = staticmethod(_idempotent_register_actor)


class ShadowAgent:
    """One PCLA agent in shadow mode — computes controls, does not apply them."""

    AGENT_CSV_FIELDS = ["tick", "sim_time_s", "throttle", "steer", "brake"]

    def __init__(
        self,
        agent_name: str,
        vehicle: carla.Actor,
        route_path: str,
        client: carla.Client,
        out_dir: str,
    ):
        self.name = agent_name
        self.pcla: PCLA | None = None
        self.csv_path = os.path.join(out_dir, f"agent_{agent_name}.csv")
        self._csv_file = open(self.csv_path, "w", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.AGENT_CSV_FIELDS)
        self._writer.writeheader()
        self.ticks_ok = 0
        self.ticks_none = 0
        self.ticks_error = 0
        self._first_error_logged = False
        try:
            self.pcla = PCLA(agent_name, vehicle, route_path, client)
            print(f"[INFO]   · Shadow agent '{agent_name}' attached")
        except Exception:
            print(f"[ERROR] Failed to initialise shadow agent '{agent_name}':")
            print(traceback.format_exc())
            self.pcla = None

    def tick(self, tick_idx: int, sim_time_s: float):
        """Call get_action(), log the result. Never applies it to the vehicle."""
        if self.pcla is None:
            return
        try:
            ctrl = self.pcla.get_action()
        except Exception:
            self.ticks_error += 1
            if not self._first_error_logged:
                print(f"[WARN] {self.name}.get_action() raised at tick {tick_idx}:")
                print(traceback.format_exc())
                self._first_error_logged = True
            return
        if ctrl is None:
            self.ticks_none += 1
            return
        self.ticks_ok += 1
        self._writer.writerow(
            {
                "tick": tick_idx,
                "sim_time_s": round(sim_time_s, 3),
                "throttle": round(ctrl.throttle, 4),
                "steer": round(ctrl.steer, 4),
                "brake": round(ctrl.brake, 4),
            }
        )

    def cleanup(self):
        print(
            f"[STATS] {self.name}: ok={self.ticks_ok} "
            f"none={self.ticks_none} error={self.ticks_error}"
        )
        try:
            self._csv_file.close()
        except Exception:
            pass
        if self.pcla is not None:
            try:
                self.pcla.cleanup()
            except Exception:
                print(f"[WARN] cleanup failed for {self.name}:")
                print(traceback.format_exc())


class ShadowAgentSet:
    """Convenience wrapper to manage multiple shadow agents on one vehicle."""

    def __init__(
        self,
        agent_names: list[str],
        vehicle: carla.Actor,
        route_path: str,
        client: carla.Client,
        out_dir: str,
    ):
        self.agents: list[ShadowAgent] = []
        for name in agent_names:
            self.agents.append(ShadowAgent(name, vehicle, route_path, client, out_dir))

    def tick(self, tick_idx: int, sim_time_s: float):
        for a in self.agents:
            a.tick(tick_idx, sim_time_s)

    def cleanup(self):
        for a in self.agents:
            a.cleanup()
