"""PHX v2 Deterministic Dependency Graph & Constraint Solver (FR-14).

Resolves transitive `requires -> provides` dependencies via DAG topological ordering,
enforces `conflicts_with` wildcard exclusions deterministically before codegen, and
orders middlewares by `middleware_order_priority`.
"""

from __future__ import annotations

import fnmatch
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from phoenix.framework.agent.capsules.registry import CapsuleRegistry
from phoenix.framework.agent.capsules.schema import CodeCapsule


@dataclass
class SolveResult:
    success: bool
    ordered_capsules: List[CodeCapsule] = field(default_factory=list)
    middleware_order: List[str] = field(default_factory=list)
    auto_added_dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)


class CapsuleConstraintSolver:
    """FR-14: Builds a verified DAG of compatible Code Capsules."""

    @staticmethod
    def _matches_conflict(cap_a: CodeCapsule, cap_b: CodeCapsule) -> bool:
        for pat in cap_a.contract.conflicts_with:
            if fnmatch.fnmatch(cap_b.capsule_id, pat):
                return True
        for pat in cap_b.contract.conflicts_with:
            if fnmatch.fnmatch(cap_a.capsule_id, pat):
                return True
        return False

    @classmethod
    def solve(
        cls,
        registry: CapsuleRegistry,
        requested_capsule_ids: Sequence[str],
        *,
        language: str = "python",
        framework: Optional[str] = None,
        orm: Optional[str] = None,
    ) -> SolveResult:
        selected: Dict[str, CodeCapsule] = {}
        auto_added: List[str] = []
        missing_reqs: List[str] = []
        conflicts: List[str] = []
        alternatives: List[str] = []

        # 1. Seed explicitly requested capsules
        queue: deque[CodeCapsule] = deque()
        for cid in requested_capsule_ids:
            cap = registry.get(cid)
            if cap is None:
                missing_reqs.append(f"Requested capsule '{cid}' not found in registry.")
                continue
            if cap.capsule_id not in selected:
                selected[cap.capsule_id] = cap
                queue.append(cap)

        if missing_reqs:
            return SolveResult(success=False, missing_requirements=missing_reqs)

        # 2. Check pairwise conflicts on explicitly requested capsules first
        sel_list = list(selected.values())
        for i in range(len(sel_list)):
            for j in range(i + 1, len(sel_list)):
                if cls._matches_conflict(sel_list[i], sel_list[j]):
                    conflicts.append(
                        f"Conflict detected: '{sel_list[i].capsule_id}' conflicts_with '{sel_list[j].capsule_id}'."
                    )
                    # Suggest compatible alternative providers if available
                    for prov in sel_list[j].contract.provides:
                        for alt in registry.find_providers(prov, language=language, framework=framework, orm=orm):
                            if alt.capsule_id != sel_list[j].capsule_id and not cls._matches_conflict(sel_list[i], alt):
                                if alt.capsule_id not in alternatives:
                                    alternatives.append(alt.capsule_id)
        if conflicts:
            return SolveResult(success=False, conflicts=conflicts, suggested_alternatives=alternatives)

        # 3. Transitive dependency closure (`requires` -> `provides`)
        edges: Dict[str, Set[str]] = {cid: set() for cid in selected}  # dep_id -> dependent_ids
        in_degree: Dict[str, int] = {cid: 0 for cid in selected}

        while queue:
            cur = queue.popleft()
            for req in cur.contract.requires:
                # Check if already satisfied by a selected capsule
                existing_provider = next(
                    (c for c in selected.values() if req in c.contract.provides),
                    None,
                )
                if existing_provider is None:
                    providers = registry.find_providers(req, language=language, framework=framework, orm=orm)
                    compatible = [
                        p for p in providers
                        if all(not cls._matches_conflict(p, s) for s in selected.values())
                    ]
                    if not compatible:
                        if providers:
                            conflicts.append(
                                f"Requirement '{req}' of '{cur.capsule_id}' only has conflicting providers: "
                                f"{[p.capsule_id for p in providers]}."
                            )
                        else:
                            missing_reqs.append(f"Missing provider for requirement '{req}' of '{cur.capsule_id}'.")
                        continue
                    # Pick highest-priority compatible provider
                    compatible.sort(key=lambda c: (c.contract.middleware_order_priority, c.capsule_id))
                    existing_provider = compatible[0]
                    selected[existing_provider.capsule_id] = existing_provider
                    edges.setdefault(existing_provider.capsule_id, set())
                    in_degree.setdefault(existing_provider.capsule_id, 0)
                    auto_added.append(existing_provider.capsule_id)
                    queue.append(existing_provider)

                if existing_provider.capsule_id != cur.capsule_id:
                    if cur.capsule_id not in edges.setdefault(existing_provider.capsule_id, set()):
                        edges[existing_provider.capsule_id].add(cur.capsule_id)
                        in_degree[cur.capsule_id] = in_degree.get(cur.capsule_id, 0) + 1

        if conflicts or missing_reqs:
            return SolveResult(
                success=False,
                conflicts=conflicts,
                missing_requirements=missing_reqs,
                suggested_alternatives=alternatives,
            )

        # 4. Deterministic Kahn topological sort tie-broken by middleware_order_priority
        ready: List[Tuple[int, str]] = sorted(
            (selected[cid].contract.middleware_order_priority, cid)
            for cid, deg in in_degree.items()
            if deg == 0
        )
        ordered: List[CodeCapsule] = []
        while ready:
            _, cid = ready.pop(0)
            ordered.append(selected[cid])
            for nxt in sorted(edges.get(cid, ())):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    ready.append((selected[nxt].contract.middleware_order_priority, nxt))
                    ready.sort()

        if len(ordered) != len(selected):
            return SolveResult(
                success=False,
                conflicts=["Cyclic dependency detected among capsules: " + ", ".join(sorted(selected.keys()))],
            )

        mw_order = [
            c.capsule_id
            for c in sorted(ordered, key=lambda c: (c.contract.middleware_order_priority, c.capsule_id))
            if "app.middlewares" in c.ast_wiring.injection_points
        ]
        return SolveResult(
            success=True,
            ordered_capsules=ordered,
            middleware_order=mw_order,
            auto_added_dependencies=auto_added,
        )
