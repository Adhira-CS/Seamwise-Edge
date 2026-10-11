"""SeamWise Edge: deterministic, traffic-level network simulation.

No sockets, real codecs, radio handovers, or measured bandwidth are used.
Controllers receive noisy synthetic observations, never a future trace.
The engine and CSV/ZIP exports use only the Python standard library.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

PATHS = ("Wi-Fi", "5G", "Satellite")
SCENARIOS = {
    "gradual": "Gradual fade: full video to reduced video, audio, text, outage, recovery",
    "abrupt": "Abrupt terrestrial failures, constrained satellite, outage, recovery",
    "correlated": "Shared terrestrial interruptions plus complete outage",
    "long_delay": "Gradual fade with long-delay satellite (650 ms one way)",
}
POLICIES = {
    "reactive": "Reactive switching (fixed full video)",
    "predictive": "Predictive switching (fixed full video)",
    "protected": "Predictive + protection (fixed full video)",
    "adaptive_reactive": "Adaptive reactive baseline",
    "seamwise": "SeamWise Edge",
}
MODE_NAMES = {0: "Offline", 1: "Text + status", 2: "Audio + status",
              3: "Reduced video", 4: "Full video"}
MODE_RATES = {0: 0.0, 1: 0.016, 2: 0.080, 3: 0.400, 4: 1.500}
HEADER_BYTES = 48  # synthetic UDP/IP + application-envelope allowance
STATUS_BYTES = 128
STATUS_INTERVAL = 0.5
STATUS_DEADLINE = 2.0
MEDIA_DEADLINE = 0.5
LOSS_LIMIT = 0.12
PROBE_BYTES = 64 + HEADER_BYTES  # request plus response per path per tick
ACK_BYTES = 32 + HEADER_BYTES
VERSION = "1.0.0"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    scenario: str = "gradual"
    duration_s: float = 100.0
    dt_s: float = 0.25
    prediction_horizon_s: float = 2.0
    duplicate_cap: float = 0.15

    def __post_init__(self):
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {self.scenario}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.dt_s != 0.25:
            raise ValueError("This model uses a fixed 250 ms control tick")
        if not 1 <= self.duration_s <= 100 or not math.isclose(
                self.duration_s / self.dt_s, round(self.duration_s / self.dt_s)):
            raise ValueError("duration_s must be 1..100 seconds in 0.25-second steps")
        if not 0 <= self.prediction_horizon_s <= 5:
            raise ValueError("prediction_horizon_s must be 0..5 seconds")
        if not 0 <= self.duplicate_cap <= 0.15:
            raise ValueError("duplicate_cap must be 0..0.15")


@dataclass
class Result:
    config: dict
    policy: str
    label: str
    summary: dict
    timeline: list[dict]
    network: list[dict]
    messages: list[dict]
    events: list[dict]
    transmissions: list[dict]


def uniform(seed: int, *keys: Any) -> float:
    """Stateless common random numbers; independent of controller call order."""
    raw = "|".join(map(str, (seed,) + keys)).encode("utf-8")
    integer = int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")
    return (integer + 0.5) / (2**64)


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def interpolate(t: float, points: list[tuple[float, float]]) -> float:
    for (ta, a), (tb, b) in zip(points, points[1:]):
        if ta <= t < tb:
            return a + (b - a) * (t - ta) / (tb - ta)
    return points[0][1] if t < points[0][0] else points[-1][1]


def make_trace(config: Config) -> list[dict]:
    """Generate immutable-by-convention shared conditions and noisy observations.

    Observations are an idealized measurement service, available at the start
    of a tick. Their control-byte allowance is charged, but the estimator is
    NOT inferred from those packets. This is a simulator limitation.
    """
    rows = []
    for tick in range(round(config.duration_s / config.dt_s)):
        t = tick * config.dt_s
        wifi = interpolate(t, [(0, 5), (15, 5), (24, 1.5), (32, 0), (100, 0)])
        mobile = interpolate(t, [(0, 3), (35, 3), (47, .6), (50, .6),
                                 (60, 0), (89.75, 0), (90, 3), (100, 3)])
        satellite = interpolate(t, [(0, .16), (65, .16), (68, .024), (100, .024)])
        if config.scenario == "abrupt":
            wifi = 5.0 if t < 23 else 0.0
            mobile = 3.0 if t < 44 or t >= 90 else (.6 if t < 55 else 0.0)
        if config.scenario == "correlated" and (26 <= t < 32 or 43 <= t < 49):
            wifi = mobile = 0.0
        capacities = (wifi, mobile, satellite)
        for p, base in zip(PATHS, capacities):
            outage = 80 <= t < 90
            available = base > 0.002 and not outage
            noise = 0.97 + 0.06 * uniform(config.seed, "capacity", tick, p)
            cap = base * noise if available else 0.0
            base_delay = {"Wi-Fi": 15.0, "5G": 25.0,
                          "Satellite": 650.0 if config.scenario == "long_delay" else 80.0}[p]
            # Deterioration increases delay/loss; values are illustrative only.
            normal = {"Wi-Fi": 5.0, "5G": 3.0, "Satellite": .16}[p]
            stress = clamp(1 - base / normal, 0, 1)
            delay = base_delay + 70 * stress + 8 * uniform(config.seed, "delay", tick, p)
            loss = clamp(.005 + .14 * stress**2 + .006 * uniform(config.seed, "loss", tick, p), 0, .4)
            if p == "Satellite":
                loss = .015 + .025 * stress  # constrained capacity need not mean a broken path
            if not available:
                loss = 1.0
            observed_cap = cap * (.96 + .08 * uniform(config.seed, "observe", tick, p))
            observed_loss = clamp(loss + .01 * (uniform(config.seed, "observe-loss", tick, p) - .5), 0, 1)
            rows.append({"time_s": t, "path": p, "capacity_mbps": cap,
                         "latency_ms": delay, "loss_pct": loss * 100,
                         "available": available,
                         "observed_capacity_mbps": observed_cap,
                         "observed_loss_pct": observed_loss * 100,
                         "observed_latency_ms": delay * (.97 + .06 * uniform(config.seed, "observe-delay", tick, p))})
    return rows


def forecast(history: list[dict], horizon: float) -> dict:
    """EWMA level plus a least-squares trend from <=3 s of observed history."""
    last = history[-1]
    if len(history) < 4 or not last["available"]:
        return {"capacity_mbps": last["observed_capacity_mbps"],
                "loss_pct": last["observed_loss_pct"],
                "latency_ms": last["observed_latency_ms"], "available": last["available"]}
    times = [x["time_s"] for x in history]
    center = mean(times)
    denominator = sum((t - center) ** 2 for t in times)
    out = {"available": last["available"]}
    for source, target, bounds in [
        ("observed_capacity_mbps", "capacity_mbps", (0, 20)),
        ("observed_loss_pct", "loss_pct", (0, 100)),
        ("observed_latency_ms", "latency_ms", (0, 2000)),
    ]:
        values = [r[source] for r in history]
        level = values[0]
        for val in values[1:]:
            level = .55 * val + .45 * level
        avg = mean(values)
        slope = sum((t - center) * (v - avg) for t, v in zip(times, values)) / denominator
        # Trend uses current measured level to avoid hiding a fast degradation.
        estimate = .5 * level + .5 * values[-1] + slope * horizon
        out[target] = clamp(estimate, *bounds)
    return out


def measured(row: dict) -> dict:
    return {"capacity_mbps": row["observed_capacity_mbps"],
            "loss_pct": row["observed_loss_pct"],
            "latency_ms": row["observed_latency_ms"], "available": row["available"]}


def admissible_mode(observation: dict) -> int:
    if not observation["available"] or observation["loss_pct"] >= 70:
        return 0
    for mode in (4, 3, 2, 1):
        deadline = STATUS_DEADLINE if mode == 1 else MEDIA_DEADLINE
        if (observation["capacity_mbps"] >= MODE_RATES[mode] * 1.25
                and observation["latency_ms"] / 1000 + .10 < deadline
                and (mode == 1 or observation["loss_pct"] / 100 <= LOSS_LIMIT)):
            return mode
    return 0


class Controller:
    """A causal path-and-mode controller, shared stability logic across policies."""
    def __init__(self, config: Config, policy: str):
        self.config = config
        self.policy = policy
        self.predictive = policy in ("predictive", "protected", "seamwise")
        self.adaptive = policy in ("adaptive_reactive", "seamwise")
        self.protected = policy in ("protected", "adaptive_reactive", "seamwise")
        self.primary: str | None = None
        self.mode = 0
        self.last_switch = -100.0
        self.risk_since: float | None = None
        self.upgrade_since: float | None = None
        self.upgrade_target: tuple | None = None
        self.protect_until = -1.0
        self.protect_reason = ""

    def step(self, t: float, current: dict[str, dict], future: dict[str, dict]) -> dict:
        # This method has no scenario, seed, or future-trace access.
        observed = {p: measured(current[p]) for p in PATHS}
        outlook = future if self.predictive else observed
        now_modes = {p: admissible_mode(observed[p]) for p in PATHS}
        future_modes = {p: min(now_modes[p], admissible_mode(outlook[p])) for p in PATHS}
        def rank(p):
            return (future_modes[p], -outlook[p]["latency_ms"],
                    -outlook[p]["loss_pct"], outlook[p]["capacity_mbps"])
        candidates = [p for p in PATHS if now_modes[p] > 0]
        best = max(candidates, key=rank) if candidates else None
        old_path, old_mode = self.primary, self.mode
        hard = self.primary is not None and now_modes[self.primary] == 0
        reason = "Hold: current path and mode remain usable"
        danger = False
        if self.primary is not None and self.mode:
            danger = future_modes[self.primary] < self.mode
        if danger:
            reason = ("Forecast risk: requested mode may lose admission headroom"
                      if self.predictive else "Measured risk: requested mode has lost admission headroom")
        self.risk_since = (t if self.risk_since is None else self.risk_since) if danger else None
        persistent = self.risk_since is not None and t - self.risk_since >= .5 - 1e-9

        if best is None:
            self.primary, self.mode = None, 0
            self.upgrade_since = None
            reason = "Offline: no observed path meets the text/status budget"
        elif self.primary is None or hard:
            self.primary = best
            # An initialized/recovered path starts conservatively; shared by both baselines.
            self.mode = (min(2, now_modes[best]) if self.adaptive else 4)
            self.last_switch = t
            self.upgrade_since = None
            reason = "Measured fallback: validated path available" if hard else "Connect: validated observed path"
        else:
            better_mode = future_modes[best] > future_modes[self.primary]
            same_mode_better = (future_modes[best] == future_modes[self.primary]
                                and outlook[best]["loss_pct"] + 6 < outlook[self.primary]["loss_pct"])
            if (best != self.primary and (better_mode or same_mode_better)
                    and (persistent or now_modes[self.primary] < self.mode)
                    and t - self.last_switch >= 3):
                self.primary = best
                self.last_switch = t
                self.risk_since = None
                self.upgrade_since = None
                reason = ("Predict: 2nd path validated before forecast deterioration"
                          if self.predictive else "React: current quality insufficient; validated alternative")
            if self.adaptive:
                target = min(now_modes[self.primary], future_modes[self.primary])
                if target < self.mode and (persistent or self.primary != old_path):
                    self.mode = max(1, target)  # keep minimum status when present evidence permits
                    self.upgrade_since = None
                    reason += "; downgrade demand after risk check"
                elif target > self.mode:
                    upgrade_target = (self.primary, target)
                    if self.upgrade_since is None or self.upgrade_target != upgrade_target:
                        self.upgrade_since, self.upgrade_target = t, upgrade_target
                    elif t - self.upgrade_since >= 5:
                        self.mode = target
                        self.upgrade_since = None
                        reason += "; upgrade after 5 s stable headroom"
                else:
                    self.upgrade_since = None

        switched = old_path is not None and self.primary != old_path
        path_change = self.primary != old_path
        # Forecast risk or switching triggers SeamWise protection. Reactive policy
        # uses measured loss (>=3%) or current observed mode failure, never outlook.
        if self.primary and self.protected:
            if self.policy == "adaptive_reactive":
                trigger = (observed[self.primary]["loss_pct"] >= 3
                           or (old_path is not None and now_modes[old_path] < old_mode))
                protection_reason = "Measured loss/current quality triggered critical protection"
            else:
                trigger = danger or switched
                protection_reason = "Forecast risk/transition triggered critical protection"
            # Each episode <=1 s, with a >=1 s gap before another episode.
            if trigger and t >= self.protect_until + 1 and len(candidates) >= 2:
                self.protect_until = t + 1.0
                self.protect_reason = protection_reason
        protecting = bool(self.primary and t < self.protect_until and len(candidates) >= 2)
        secondary = None
        if protecting:
            alternatives = [p for p in candidates if p != self.primary]
            secondary = max(alternatives, key=rank) if alternatives else None
        return {"primary": self.primary, "mode": self.mode, "reason": reason,
                "protecting": protecting, "secondary": secondary,
                "protection_reason": self.protect_reason if protecting else "",
                "path_changed": path_change, "mode_changed": old_mode != self.mode,
                "old_path": old_path, "old_mode": old_mode}


def _chunks(total: int, maximum: int = 1000) -> list[int]:
    chunks = [maximum] * (total // maximum)
    if total % maximum:
        chunks.append(total % maximum)
    return chunks


def run_simulation(config: Config, policy: str, trace: list[dict] | None = None) -> Result:
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    trace = make_trace(config) if trace is None else trace
    ticks = round(config.duration_s / config.dt_s)
    if len(trace) != ticks * len(PATHS):
        raise ValueError("Trace length does not match Config")
    controller = Controller(config, policy)
    histories = {p: [] for p in PATHS}
    timeline, network, events, messages, transmissions = [], [], [], [], []
    totals = {k: 0 for k in ("original_payload_bytes", "duplicate_payload_bytes", "wire_bytes",
                            "probe_wire_bytes", "ack_wire_bytes", "header_wire_bytes", "suppressed_copies")}
    previous_protecting = False
    for tick in range(ticks):
        t = tick * config.dt_s
        conditions = {r["path"]: r for r in trace[tick * 3:(tick + 1) * 3]}
        future = {}
        for p in PATHS:
            h = histories[p]
            h.append(conditions[p])
            while h and h[0]["time_s"] < t - 3 + 1e-9:
                h.pop(0)
            future[p] = forecast(h, config.prediction_horizon_s)
            network.append({**conditions[p], "predicted_capacity_mbps": future[p]["capacity_mbps"]})
        decision = controller.step(t, conditions, future)
        primary, mode = decision["primary"], decision["mode"]
        if decision["path_changed"] or decision["mode_changed"]:
            events.append({"time_s": t, "event": "path/mode", "old_path": decision["old_path"] or "Offline",
                           "new_path": primary or "Offline", "old_mode": decision["old_mode"],
                           "new_mode": mode, "reason": decision["reason"]})
        if decision["protecting"] and not previous_protecting:
            events.append({"time_s": t, "event": "protect", "old_path": primary,
                           "new_path": decision["secondary"], "old_mode": mode,
                           "new_mode": mode, "reason": decision["protection_reason"]})
        previous_protecting = decision["protecting"]

        # The service slot is bounded to this epoch (no backlog carried forward).
        # One request + response probe allowance takes forward service capacity;
        # this conservatively treats each path as one shared duplex byte budget.
        available_bytes = {p: int(conditions[p]["capacity_mbps"] * 1e6 * config.dt_s / 8) for p in PATHS}
        used = {p: 0 for p in PATHS}
        for p in PATHS:
            probe = PROBE_BYTES * (2 if conditions[p]["available"] else 1)
            used[p] += probe
            totals["wire_bytes"] += probe
            totals["probe_wire_bytes"] += probe

        msg = None
        if tick % 2 == 0:
            path_available = any(
                c["available"] and c["capacity_mbps"] > 0
                and c["latency_ms"] / 1000 + (STATUS_BYTES + HEADER_BYTES) * 8 / (c["capacity_mbps"] * 1e6) <= STATUS_DEADLINE
                for c in conditions.values())
            msg = {"message_id": len(messages), "generated_s": t,
                   "deadline_s": t + STATUS_DEADLINE, "path_available": path_available,
                   "first_arrival_s": None, "latency_ms": None, "on_time": False,
                   "copies_sent": 0, "copies_received": 0, "deduplicated_copies": 0, "first_path": ""}
            messages.append(msg)

        # Fixed full-video ablations still send low-priority video into constrained
        # slots; common scheduler gives critical traffic precedence for every policy.
        packets = []
        if primary and mode:
            payload_budget = int(MODE_RATES[mode] * 1e6 * config.dt_s / 8)
            if msg is not None:
                packets.append(("critical", msg["message_id"], STATUS_BYTES, STATUS_DEADLINE))
            remaining = max(0, payload_budget - (STATUS_BYTES if msg is not None else 0))
            if mode >= 2:
                audio_bytes = min(750, remaining)  # 24 kbps payload model
                for j, size in enumerate(_chunks(audio_bytes, 250)):
                    packets.append(("audio", tick * 1000 + j, size, MEDIA_DEADLINE))
                remaining -= audio_bytes
            stream = "video" if mode >= 3 else "text"
            for j, size in enumerate(_chunks(remaining)):
                packets.append((stream, tick * 1000 + j, size,
                                MEDIA_DEADLINE if stream == "video" else STATUS_DEADLINE))

        epoch_original = epoch_duplicate = 0
        arrivals_audio = arrivals_video = offered_audio = offered_video = 0

        def transmit(path, stream, sequence, size, deadline, duplicate=False):
            nonlocal epoch_original, epoch_duplicate
            c = conditions[path]
            wire = size + HEADER_BYTES
            # Reserve room for a modeled per-critical-copy ACK, even if loss later
            # means no ACK is emitted. ACK cost is counted only for received copies.
            reserve = ACK_BYTES if stream == "critical" else 0
            row = {"time_s": t, "arrival_s": None, "path": path, "stream": stream,
                   "sequence": sequence, "payload_bytes": size, "duplicate": duplicate,
                   "outcome": "queue_drop", "wire_bytes": 0}
            if not c["available"]:
                row["outcome"] = "offline"
            elif used[path] + wire + reserve <= available_bytes[path]:
                used[path] += wire + reserve
                if duplicate:
                    totals["duplicate_payload_bytes"] += size
                    epoch_duplicate += size
                else:
                    totals["original_payload_bytes"] += size
                    epoch_original += size
                totals["wire_bytes"] += wire
                totals["header_wire_bytes"] += HEADER_BYTES
                row["wire_bytes"] = wire
                if stream == "critical":
                    messages[sequence]["copies_sent"] += 1
                lost = uniform(config.seed, "packet-loss", tick, path, stream, sequence) < c["loss_pct"] / 100
                if lost:
                    row["outcome"] = "lost"
                else:
                    jitter_s = .004 * uniform(config.seed, "packet-jitter", tick, path, stream, sequence)
                    serialization = used[path] * 8 / (c["capacity_mbps"] * 1e6)
                    arrival = t + serialization + c["latency_ms"] / 1000 + jitter_s
                    row["arrival_s"] = arrival
                    row["outcome"] = "delivered" if arrival <= t + deadline + 1e-9 else "late"
                    if stream == "critical":
                        totals["wire_bytes"] += ACK_BYTES
                        totals["ack_wire_bytes"] += ACK_BYTES
            transmissions.append(row)
            return row

        for stream, sequence, size, deadline in packets:
            row = transmit(primary, stream, sequence, size, deadline)
            if stream == "audio":
                offered_audio += size
                arrivals_audio += size if row["outcome"] == "delivered" else 0
            elif stream == "video":
                offered_video += size
                arrivals_video += size if row["outcome"] == "delivered" else 0

        # Copies are scheduled after originals so the cap uses actual original
        # transmitted payload, not offered bytes or previous saved credits.
        if decision["protecting"] and msg is not None and decision["secondary"]:
            if msg["copies_sent"] and epoch_duplicate + STATUS_BYTES <= config.duplicate_cap * epoch_original + 1e-9:
                transmit(decision["secondary"], "critical", msg["message_id"], STATUS_BYTES, STATUS_DEADLINE, True)
            else:
                totals["suppressed_copies"] += 1

        timeline.append({"time_s": t, "primary": primary or "Offline", "mode": mode,
                         "mode_name": MODE_NAMES[mode],
                         "predicted_capacity_mbps": future[primary]["capacity_mbps"] if primary else 0,
                         "observed_capacity_mbps": conditions[primary]["observed_capacity_mbps"] if primary else 0,
                         "protecting": decision["protecting"], "reason": decision["reason"],
                         "status_age_s": None, "mission_fresh": False,
                         # Generation-epoch success scored from actual modeled arrivals;
                         # these are retrospective interval outcomes, not live playback.
                         "audio_usable": offered_audio > 0 and arrivals_audio >= .8 * offered_audio,
                         "video_usable": offered_video > 0 and arrivals_video >= .8 * offered_video,
                         "duplicate_payload_bytes": totals["duplicate_payload_bytes"],
                         "original_payload_bytes": totals["original_payload_bytes"],
                         "wire_bytes": totals["wire_bytes"], "critical_delivered": 0})

    # Receiver ledger: first timely copy delivers once, later timely copies dedup.
    critical_copies = [x for x in transmissions if x["stream"] == "critical" and x["arrival_s"] is not None]
    critical_copies.sort(key=lambda x: x["arrival_s"])
    deliveries = []
    late_critical_copies = 0
    for copy in critical_copies:
        msg = messages[copy["sequence"]]
        msg["copies_received"] += 1
        if copy["arrival_s"] > msg["deadline_s"] + 1e-9:
            late_critical_copies += 1
            continue
        if msg["on_time"]:
            msg["deduplicated_copies"] += 1
        else:
            msg.update({"on_time": True, "first_arrival_s": copy["arrival_s"],
                        "latency_ms": (copy["arrival_s"] - msg["generated_s"]) * 1000,
                        "first_path": copy["path"]})
            deliveries.append((copy["arrival_s"], msg["generated_s"]))
    deliveries.sort()

    def receiver_at(times):
        latest, index = None, 0
        for t in times:
            while index < len(deliveries) and deliveries[index][0] <= t + 1e-9:
                generated = deliveries[index][1]
                latest = generated if latest is None else max(latest, generated)
                index += 1
            age = None if latest is None else t - latest
            yield age, age is not None and age <= STATUS_DEADLINE + 1e-9, index

    for row, (age, fresh, count) in zip(timeline, receiver_at(x["time_s"] for x in timeline)):
        row.update({"status_age_s": age, "mission_fresh": fresh, "critical_delivered": count})
    sample_times = [i / 10 for i in range(math.ceil(config.duration_s * 10))]
    fresh_samples = sum(fresh for _, fresh, _ in receiver_at(sample_times))
    on_time = sum(m["on_time"] for m in messages)
    available_messages = [m for m in messages if m["path_available"]]
    available_on_time = sum(m["on_time"] for m in available_messages)
    summary = {"policy": policy, "label": POLICIES[policy], "seed": config.seed,
               "scenario": config.scenario, "duration_s": config.duration_s,
               "critical_generated": len(messages), "critical_on_time": on_time,
               "critical_success_pct": 100 * on_time / len(messages) if messages else 0,
               "available_critical_generated": len(available_messages),
               "available_critical_on_time": available_on_time,
               "available_critical_success_pct": 100 * available_on_time / len(available_messages) if available_messages else None,
               "mission_continuity_pct": 100 * fresh_samples / len(sample_times),
               "audio_usable_s": sum(r["audio_usable"] for r in timeline) * config.dt_s,
               "video_usable_s": sum(r["video_usable"] for r in timeline) * config.dt_s,
               "full_video_usable_s": sum(r["video_usable"] and r["mode"] == 4 for r in timeline) * config.dt_s,
               "reduced_video_usable_s": sum(r["video_usable"] and r["mode"] == 3 for r in timeline) * config.dt_s,
               "text_mode_s": sum(r["mode"] == 1 for r in timeline) * config.dt_s,
               "offline_mode_s": sum(r["mode"] == 0 for r in timeline) * config.dt_s,
               "switches": sum(e["event"] == "path/mode" and e["old_path"] != "Offline"
                               and e["new_path"] != "Offline" and e["old_path"] != e["new_path"] for e in events),
               **totals,
               "duplicate_overhead_pct": 100 * totals["duplicate_payload_bytes"] / totals["original_payload_bytes"] if totals["original_payload_bytes"] else 0,
               "deduplicated_copies": sum(m["deduplicated_copies"] for m in messages),
               "late_critical_copies": late_critical_copies}
    return Result(asdict(config), policy, POLICIES[policy], summary, timeline, network, messages, events, transmissions)


def run_comparison(config: Config, policies=None) -> dict[str, Result]:
    policies = ("adaptive_reactive", "seamwise") if policies is None else tuple(policies)
    if not policies:
        raise ValueError("Choose at least one policy")
    trace = make_trace(config)
    return {p: run_simulation(config, p, trace) for p in policies}


def csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def export_files(results: dict[str, Result]) -> dict[str, str]:
    files = {"summary.csv": csv_text([r.summary for r in results.values()])}
    metadata = {"simulation_version": VERSION, "scope": "synthetic traffic-level simulation; no real media or hardware",
                "deadline_seconds": {"critical": STATUS_DEADLINE, "media": MEDIA_DEADLINE},
                "mission_sampling_s": .1, "media_interval_success_fraction": .8,
                "observations": "idealized noisy current observations, not packet-derived bandwidth estimates",
                "randomness": "stateless common uniforms indexed by seed/tick/path/stream/sequence",
                "service_model": "per-path bounded 250ms service slots; critical first; no carried queue",
                "mode_budget_mbps": MODE_RATES,
                "configs": {p: r.config for p, r in results.items()}}
    files["metadata.json"] = json.dumps(metadata, indent=2)
    for key, result in results.items():
        for field in ("timeline", "network", "messages", "events", "transmissions"):
            files[f"{key}/{field}.csv"] = csv_text(getattr(result, field))
    return files


def result_zip(results: dict[str, Result]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as z:
        for name, content in export_files(results).items():
            z.writestr(name, content)
    return buffer.getvalue()


def write_results(results: dict[str, Result], output_dir) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, content in export_files(results).items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")


if __name__ == "__main__":
    results = run_comparison(Config())
    for result in results.values():
        print(json.dumps(result.summary, indent=2))
