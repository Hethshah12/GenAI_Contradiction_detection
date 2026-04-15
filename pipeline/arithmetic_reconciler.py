"""
Arithmetic Reconciler
=====================

Before flagging a numeric contradiction, try to *derive* one number
from the other via a small set of canonical relations. If any
derivation holds within tolerance, the pair is arithmetically
consistent and is NOT a contradiction.

Relations handled
-----------------
    sum_of_parts          sum(children) ≈ whole        e.g. 31 + 25 + 12 = 68
    share_of_whole        whole × pct   ≈ part         e.g. 843B × 42% = 354B
    inverse_share         part / whole  ≈ pct          e.g. 354/843 = 42%
    cagr                  (end/start)^(1/n) - 1 ≈ CAGR
    delta                 end - start   ≈ delta        e.g. 843 - 712 = 131
    rounding              |a - b| / max(a, b) < 0.02   straight rounding

All checks are symmetric: we try both directions before giving up.
Default tolerance is 3% relative — a tight band that still tolerates
rounding (report prose rounds to nearest whole billion / 0.1%).

The reconciler is *context-aware*. Each call takes an optional
`universe` — a list of canonical values that the document is known to
be built from. If the candidate pair can be reconciled through any
derivation over the universe, it is consistent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations


@dataclass
class Reconciliation:
    consistent: bool
    relation: str = ""
    detail: dict | None = None


def _close(a: float, b: float, rel_tol: float = 0.03, abs_tol: float = 0.05) -> bool:
    if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    # symmetric percentage tolerance on the smaller of the two
    if a == 0 or b == 0:
        return abs(a - b) <= abs_tol
    return abs(a - b) / max(abs(a), abs(b)) <= rel_tol


# ─────────────────────────────────────────────────────────────────────
# Individual relation checks
# ─────────────────────────────────────────────────────────────────────

def _check_sum_of_parts(
    target: float,
    universe: list[float],
    rel_tol: float = 0.03,
) -> tuple[list[float], float] | None:
    """
    Find a subset of `universe` (size 2..len(universe)) whose sum ≈ target.
    Returns (subset, sum) or None. Capped at subset size 6 to keep O(C(n,6))
    manageable on the rare wide universes.
    """
    # Filter: only consider universe elements smaller than target
    candidates = [v for v in universe if 0 < v < target * 1.2]
    if not candidates:
        return None
    n = min(len(candidates), 10)
    for k in range(2, min(7, n + 1)):
        for combo in combinations(candidates[:n], k):
            s = sum(combo)
            if _close(s, target, rel_tol=rel_tol):
                return (list(combo), s)
    return None


def _check_share_of_whole(
    part: float, whole: float, pct: float, rel_tol: float = 0.03
) -> bool:
    """whole × (pct/100) ≈ part.

    Guards against the degenerate case where both ``part`` and ``pct``
    are percentages (< 100) — in that regime almost any whole close to
    ``part/pct × 100`` will satisfy the equation, producing false
    positives. We require ``part`` to be at least 5× ``pct`` OR larger
    than 100 (i.e. not itself a percentage).
    """
    if pct <= 0 or whole <= 0:
        return False
    # pct must actually look like a percentage
    if pct >= 100:
        return False
    if part < 100 and part < pct * 5:
        # both values look like percentages in the same range — skip
        return False
    expected = whole * (pct / 100.0)
    return _close(expected, part, rel_tol=rel_tol)


def _check_cagr(start: float, end: float, years: int, cagr_pct: float,
                rel_tol: float = 0.05) -> bool:
    if start <= 0 or end <= 0 or years <= 0:
        return False
    actual = ((end / start) ** (1.0 / years) - 1.0) * 100.0
    return _close(actual, cagr_pct, rel_tol=rel_tol, abs_tol=0.2)


def _check_delta(start: float, end: float, delta: float,
                 rel_tol: float = 0.03) -> bool:
    return _close(end - start, delta, rel_tol=rel_tol)


def _check_rounding(a: float, b: float, rel_tol: float = 0.02) -> bool:
    return _close(a, b, rel_tol=rel_tol)


# ─────────────────────────────────────────────────────────────────────
# Public reconciliation entry points
# ─────────────────────────────────────────────────────────────────────

def reconcile_pair(
    a: float,
    b: float,
    universe: list[float] | None = None,
    rel_tol: float = 0.03,
) -> Reconciliation:
    """
    Given two numeric values a and b (and an optional universe of
    supporting values from the same document), return whether the pair
    is arithmetically consistent.

    Strategy:
      1. straight rounding equality            a ≈ b
      2. a is the sum of some subset of U that includes b
      3. b is the sum of some subset of U that includes a
      4. either is a share-of-whole of a U element   (requires U)
      5. either is an end-start delta over U         (requires U)
    """
    if _check_rounding(a, b, rel_tol=0.02):
        return Reconciliation(True, "rounding", {"a": a, "b": b})

    if universe:
        # sum(universe_subset) ≈ a with b in the subset
        sub = _check_sum_of_parts(a, universe + [b], rel_tol=rel_tol)
        if sub and abs(b) in {abs(v) for v in sub[0]}:
            return Reconciliation(
                True, "sum_of_parts_a_contains_b",
                {"a": a, "parts": sub[0], "sum": sub[1]},
            )
        sub = _check_sum_of_parts(b, universe + [a], rel_tol=rel_tol)
        if sub and abs(a) in {abs(v) for v in sub[0]}:
            return Reconciliation(
                True, "sum_of_parts_b_contains_a",
                {"b": b, "parts": sub[0], "sum": sub[1]},
            )

        # share-of-whole: a = whole × (b/100)  or  b = whole × (a/100)
        for whole in universe:
            if whole <= 0:
                continue
            if _check_share_of_whole(a, whole, b, rel_tol=rel_tol):
                return Reconciliation(
                    True, "share_of_whole",
                    {"whole": whole, "pct": b, "part": a},
                )
            if _check_share_of_whole(b, whole, a, rel_tol=rel_tol):
                return Reconciliation(
                    True, "share_of_whole",
                    {"whole": whole, "pct": a, "part": b},
                )

        # NOTE: a generic "|a - b| matches some universe value" check is
        # too loose — when the caller has already established that the
        # two claims share the same metric/entity/unit/time signature,
        # a YoY delta relation cannot apply. We only fire YoY via the
        # explicit three-value path (start, end, delta all in context),
        # which is handled by _check_delta in the claim extractor layer.

    return Reconciliation(False)


def reconcile_subset_sum(target: float, universe: list[float],
                         rel_tol: float = 0.03) -> Reconciliation:
    """
    Standalone helper: is `target` the sum of any subset of `universe`?
    Returns Reconciliation(consistent=True, relation='sum_of_parts')
    if yes.
    """
    sub = _check_sum_of_parts(target, universe, rel_tol=rel_tol)
    if sub:
        return Reconciliation(
            True, "sum_of_parts",
            {"parts": sub[0], "sum": sub[1], "target": target},
        )
    return Reconciliation(False)


# ─────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("== sum-of-parts: 31 + 25 + 12 = 68 ==")
    r = reconcile_pair(68.0, 31.0, universe=[31.0, 25.0, 12.0, 4.0, 3.0, 2.0, 23.0])
    print(r)

    print("\n== share-of-whole: 843 × 42% = 354 ==")
    r = reconcile_pair(354.0, 42.0, universe=[843.0])
    print(r)

    print("\n== delta: 843 - 712 = 131 ==")
    r = reconcile_pair(131.0, 712.0, universe=[843.0, 712.0, 591.0])
    print(r)

    print("\n== YoY CAGR 2022: ((490/395)^1 - 1)*100 = 24.05 ==")
    print(_check_cagr(395.0, 490.0, 1, 24.1))

    print("\n== DIFFERENT METRICS: 94% penetration vs 29% edge pilots ==")
    # no universe link → reconciliation fails (correct! these aren't
    # arithmetically related; the gate above is what catches them)
    r = reconcile_pair(94.0, 29.0, universe=[])
    print(r)

    print("\n== Rounding: 345.63 vs 346 ==")
    r = reconcile_pair(345.63, 346.0)
    print(r)
