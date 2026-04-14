"""Shadow-mode PCLA agent runner.

Attaches multiple PCLA agents to the same follower vehicle at once. Each
agent brings its own sensor suite (attached by PCLA internally) and is
invoked every simulation tick via `get_action()`. The returned
`carla.VehicleControl` is logged to CSV but NOT applied to the vehicle —
the vehicle is driven externally by the scenario script (e.g. Traffic
Manager).

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
            print(f"[WARN] {self.name}.get_action() raised at tick {tick_idx}:")
            print(traceback.format_exc())
            return
        if ctrl is None:
            return
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
