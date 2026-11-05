"""Manual identity corrections as an ordered edit log.

Automatic tracking (plus learned joins) numbers the people in a run; the user then
fixes mistakes. Whole-track merges used to be the only fix, saved as merge groups.
The edit log adds the fixes long, crowded runs need, each replayed in order:

    {"op": "merge",  "groups": [[a, b, ...], ...]}       people become one
    {"op": "split",  "person": p, "frame": f}            p from frame f on is a new person
    {"op": "swap",   "people": [a, b], "frames": [f0, f1]}  exchange a and b in f0..f1
    {"op": "remove", "person": p, "frames": [f0, f1]}    drop p's detections in f0..f1

Person numbers in an edit are the ones on screen when it was made (after the edits
before it), so replaying the log always reproduces what the user saw. A merge
renumbers like `merge_track_ids` (merged people take their lowest member's place);
a split appends the new person at the end, so nobody else is renumbered.

A log of only merges is stored in the legacy merge-groups form (`canonical`), so
saved merges, labels and identity learning keep working unchanged; only runs that
use split/swap/remove store the edit list.
"""

from __future__ import annotations

import numpy as np

OPS = ("merge", "split", "swap", "remove")


def is_edit_log(value) -> bool:
    """True for a list of edit dicts, False for legacy merge groups (or nothing)."""
    return bool(value) and all(isinstance(e, dict) for e in value)


def _clean(edit: dict) -> dict:
    op = edit.get("op")
    if op == "merge":
        groups = sorted(sorted({int(t) for t in g}) for g in edit["groups"] if len({int(t) for t in g}) > 1)
        return {"op": "merge", "groups": groups}
    if op == "split":
        return {"op": "split", "person": int(edit["person"]), "frame": int(edit["frame"])}
    if op == "swap":
        a, b = sorted(int(p) for p in edit["people"])
        f0, f1 = sorted(int(f) for f in edit["frames"])
        if a == b:
            raise ValueError("swap needs two different people")
        return {"op": "swap", "people": [a, b], "frames": [f0, f1]}
    if op == "remove":
        f0, f1 = sorted(int(f) for f in edit["frames"])
        return {"op": "remove", "person": int(edit["person"]), "frames": [f0, f1]}
    raise ValueError(f"unknown identity edit {op!r}")


def from_saved(value) -> list[dict]:
    """Edit history for a saved value: legacy merge groups become one merge edit."""
    if not value:
        return []
    if is_edit_log(value):
        return [_clean(e) for e in value]
    return [{"op": "merge", "groups": [sorted({int(t) for t in g}) for g in value]}]


def legacy_groups(edits: list[dict], n_base: int) -> list[list[int]] | None:
    """The merge groups (in base numbering) equivalent to a merges-only log, else None."""
    members = [[t] for t in range(n_base)]
    for e in edits:
        if e["op"] != "merge":
            return None
        target = list(range(len(members)))
        for g in e["groups"]:
            if any(p < 0 or p >= len(members) for p in g):
                raise ValueError(f"merge ids {g} don't match the {len(members)} people shown")
            for p in g:
                target[p] = g[0]
        merged: dict[int, list[int]] = {}
        for p, m in enumerate(members):
            merged.setdefault(target[p], []).extend(m)
        members = [sorted(merged[k]) for k in sorted(merged)]
    return sorted(m for m in members if len(m) > 1)


def canonical(edits: list[dict], n_base: int):
    """What to save for an edit history: legacy merge groups when it's merges only."""
    edits = [_clean(e) for e in edits]
    groups = legacy_groups(edits, n_base)
    return groups if groups is not None else edits


def _compact(labels: list, kps: list) -> tuple[list, int]:
    """Renumber people 0..n-1 keeping their order; drop numbers no longer used.
    A person left with two detections in one frame keeps the more confident one."""
    used = sorted({int(t) for a in labels for t in a if t >= 0})
    renum = np.full(max(used, default=-1) + 2, -1, int)
    renum[used] = np.arange(len(used))
    out = []
    for f, a in enumerate(labels):
        b = np.where(a >= 0, renum[np.maximum(a, 0)], -1) if len(a) else a.copy()
        for tid in {t for t in b if t >= 0 and (b == t).sum() > 1}:
            dup = np.flatnonzero(b == tid)
            best = max(dup, key=lambda i: np.nanmean(kps[f][i][:, 2]))
            b[[i for i in dup if i != best]] = -1
        out.append(b)
    return out, len(used)


def apply_edits(kps: list, ids: list, edits: list[dict]) -> tuple[list, int]:
    """Replay `edits` on per-frame track ids. Returns (ids, n_people) like `stabilize`."""
    from .tracking import merge_track_ids

    labels = [np.asarray(a, int).copy() for a in ids]
    n = 1 + max((int(t) for a in labels for t in a), default=-1)
    for e in (_clean(e) for e in edits):
        if e["op"] == "merge":
            people = [p for g in e["groups"] for p in g]
        elif e["op"] == "swap":
            people = e["people"]
        else:
            people = [e["person"]]
        if any(p < 0 or p >= n for p in people):
            raise ValueError(f"{e['op']} refers to person {max(people) + 1}, "
                             f"but only {n} people are shown")
        if e["op"] == "merge":
            labels, n = merge_track_ids(kps, labels, e["groups"])
            continue
        if e["op"] == "split":
            p, f0 = e["person"], e["frame"]
            for f in range(max(f0, 0), len(labels)):
                labels[f][labels[f] == p] = n
            n += 1
        elif e["op"] == "swap":
            (a, b), (f0, f1) = e["people"], e["frames"]
            for f in range(max(f0, 0), min(f1 + 1, len(labels))):
                x = labels[f]
                ia, ib = x == a, x == b
                x[ia], x[ib] = b, a
        elif e["op"] == "remove":
            p, (f0, f1) = e["person"], e["frames"]
            for f in range(max(f0, 0), min(f1 + 1, len(labels))):
                labels[f][labels[f] == p] = -1
        labels, n = _compact(labels, kps)
    return labels, n


def apply_identity(kps: list, ids: list, value) -> tuple[list, int]:
    """Apply saved identity corrections: legacy merge groups or an edit log."""
    from .tracking import merge_track_ids

    if is_edit_log(value):
        return apply_edits(kps, ids, value)
    groups = [sorted(set(g)) for g in (value or []) if len(set(g)) > 1]
    if not groups:
        return [a.copy() for a in ids], 1 + max((int(t) for a in ids for t in a), default=-1)
    return merge_track_ids(kps, ids, groups)


def track_identity(base_ids: list, final_ids: list) -> dict[int, int | None]:
    """For each base track, the one person it ended up as -- or None if edits split it
    across people (a split/swapped track is no longer evidence about one identity)."""
    seen: dict[int, set] = {}
    for a, b in zip(base_ids, final_ids):
        for t, p in zip(a, b):
            if t >= 0:
                s = seen.setdefault(int(t), set())
                if p >= 0:
                    s.add(int(p))
    return {t: (next(iter(s)) if len(s) == 1 else None) for t, s in seen.items()}


def fragments_after(fragments: list[int], base_ids: list, final_ids: list, n: int) -> list[int]:
    """Automatic fragments behind each final person (a split track counts for both parts)."""
    parts: list[set] = [set() for _ in range(n)]
    for a, b in zip(base_ids, final_ids):
        for t, p in zip(a, b):
            if t >= 0 and p >= 0:
                parts[int(p)].add(int(t))
    return [sum(fragments[t] for t in s) if s else 1 for s in parts]
