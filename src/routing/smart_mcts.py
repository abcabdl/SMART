from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SmartMCTSNode:
    role: Optional[str]
    depth: int
    selected_ids: Tuple[str, ...]
    selected_roles: Tuple[str, ...]
    gap_state: Dict[str, Any]
    parent: Optional["SmartMCTSNode"] = None
    action_card_id: Optional[str] = None
    action_reward: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    children: Dict[str, "SmartMCTSNode"] = field(default_factory=dict)
    untried_actions: List[str] = field(default_factory=list)
    visits: int = 0
    total_return: float = 0.0

    @property
    def value(self) -> float:
        if self.visits <= 0:
            return 0.0
        return self.total_return / float(self.visits)


def _gap_count(gap_state: Dict[str, Any]) -> int:
    return int((gap_state or {}).get("gap_count") or 0)


def select_with_smart_mcts(
    *,
    initial_role: str,
    initial_selected_ids: Tuple[str, ...],
    initial_selected_roles: Tuple[str, ...],
    initial_gap_state: Dict[str, Any],
    iterations: int,
    rollout_depth: int,
    exploration: float,
    discount: float,
    max_candidates: int,
    action_ids_for_node: Callable[[str, Tuple[str, ...], Tuple[str, ...], Dict[str, Any], int], List[str]],
    score_action: Callable[[Tuple[str, ...], Tuple[str, ...], str, str, Dict[str, Any], int], Dict[str, Any]],
    predict_next_role: Callable[[Optional[str], int, Dict[str, Any]], Optional[str]],
) -> Optional[Dict[str, Any]]:
    """Run SMART's LLM-free MCTS over role-conditioned candidate IDs.

    The callbacks provide candidate construction, Eq. (6) scoring, and
    deterministic next-role prediction. This function never calls an LLM and
    never executes tools.
    """

    depth_limit = max(1, int(rollout_depth or 1))
    candidate_cap = max(1, int(max_candidates or 1))
    iterations = max(1, int(iterations or 1))
    exploration = max(0.0, float(exploration))
    discount = max(0.0, min(1.0, float(discount)))

    def init_untried(node: SmartMCTSNode) -> List[str]:
        if not node.role:
            return []
        ids = action_ids_for_node(
            node.role,
            node.selected_ids,
            node.selected_roles,
            node.gap_state,
            node.depth,
        )
        return list(ids[:candidate_cap])

    root = SmartMCTSNode(
        role=initial_role,
        depth=0,
        selected_ids=tuple(initial_selected_ids or ()),
        selected_roles=tuple(initial_selected_roles or ()),
        gap_state=dict(initial_gap_state or {}),
    )
    root.untried_actions = init_untried(root)
    if not root.untried_actions:
        return None

    def is_terminal(node: SmartMCTSNode) -> bool:
        return node.role is None or node.depth >= depth_limit or _gap_count(node.gap_state) <= 0

    for _ in range(iterations):
        node = root
        path = [root]
        total_return = 0.0

        while not is_terminal(node) and not node.untried_actions and node.children:
            best_child = None
            best_ucb = None
            parent_visits = max(1, node.visits)
            for child in node.children.values():
                if child.visits <= 0:
                    ucb = float("inf")
                else:
                    exploit = child.value
                    explore = exploration * math.sqrt(
                        (2.0 * max(1e-9, math.log(float(parent_visits)))) / float(child.visits)
                    )
                    ucb = exploit + explore
                if best_ucb is None or ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            if best_child is None:
                break
            node = best_child
            path.append(node)
            total_return += (discount ** max(0, node.depth - 1)) * node.action_reward

        if not is_terminal(node) and node.untried_actions:
            action_id = node.untried_actions.pop(0)
            metrics = score_action(
                node.selected_ids,
                node.selected_roles,
                node.role or "",
                action_id,
                node.gap_state,
                node.depth,
            )
            reward = float(metrics.get("reward", 0.0) or 0.0)
            next_gap_state = dict(metrics.get("gap_state") or {})
            next_role = predict_next_role(node.role, node.depth + 1, next_gap_state)
            child = SmartMCTSNode(
                role=next_role,
                depth=node.depth + 1,
                selected_ids=node.selected_ids + (action_id,),
                selected_roles=node.selected_roles + (str(node.role or ""),),
                gap_state=next_gap_state,
                parent=node,
                action_card_id=action_id,
                action_reward=reward,
                metrics=dict(metrics),
            )
            child.untried_actions = init_untried(child)
            node.children[action_id] = child
            node = child
            path.append(node)
            total_return += (discount ** max(0, node.depth - 1)) * reward

        sim_role = node.role
        sim_depth = node.depth
        sim_selected = list(node.selected_ids)
        sim_roles = list(node.selected_roles)
        sim_gap_state = dict(node.gap_state or {})

        while sim_role is not None and sim_depth < depth_limit and _gap_count(sim_gap_state) > 0:
            rollout_ids = action_ids_for_node(
                sim_role,
                tuple(sim_selected),
                tuple(sim_roles),
                sim_gap_state,
                sim_depth,
            )
            if not rollout_ids:
                break
            best_id = None
            best_metrics = None
            for cand_id in rollout_ids[:candidate_cap]:
                metrics = score_action(
                    tuple(sim_selected),
                    tuple(sim_roles),
                    sim_role,
                    cand_id,
                    sim_gap_state,
                    sim_depth,
                )
                if best_metrics is None or float(metrics.get("reward", 0.0)) > float(best_metrics.get("reward", 0.0)):
                    best_id = cand_id
                    best_metrics = metrics
            if best_id is None or best_metrics is None:
                break
            total_return += (discount ** sim_depth) * float(best_metrics.get("reward", 0.0) or 0.0)
            sim_selected.append(best_id)
            sim_roles.append(str(sim_role or ""))
            sim_gap_state = dict(best_metrics.get("gap_state") or sim_gap_state)
            sim_depth += 1
            sim_role = predict_next_role(sim_role, sim_depth, sim_gap_state)

        for path_node in path:
            path_node.visits += 1
            path_node.total_return += float(total_return)

    if not root.children:
        return None

    best_main = max(root.children.items(), key=lambda item: item[1].visits)[0]
    shadows = sorted(
        [(card_id, child.visits, child.value) for card_id, child in root.children.items() if card_id != best_main],
        key=lambda item: (item[1], item[2]),
        reverse=True,
    )
    children = {}
    for card_id, child in root.children.items():
        metrics = dict(child.metrics or {})
        children[card_id] = {
            "visits": child.visits,
            "avg_return": child.value,
            "avg_reward": child.value,
            "reward": float(metrics.get("reward", child.action_reward) or 0.0),
            "unary_score": float(metrics.get("unary_score", 0.5) or 0.0),
            "transition_score": float(metrics.get("transition_score", 0.5) or 0.0),
            "io_score": float(metrics.get("io_score", 0.5) or 0.0),
            "gap_reduction": float(metrics.get("gap_reduction", 0.0) or 0.0),
            "gap_state": dict(metrics.get("gap_state") or child.gap_state or {}),
            "posterior_score": float(metrics.get("transition_score", 0.5) or 0.0),
            "uncertainty_score": 0.0,
        }

    selected_shadows = [item[0] for item in shadows[:2]]
    return {
        "selected_main": best_main,
        "selected_shadows": selected_shadows,
        "selected_shadow": selected_shadows[0] if selected_shadows else None,
        "children": children,
    }
