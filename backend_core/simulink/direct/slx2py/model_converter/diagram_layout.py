"""Native-aware geometry, graph layering and obstacle-avoiding orthogonal routes."""
from __future__ import annotations

from collections import defaultdict, deque
import heapq
import math

SIDES = ("right", "bottom", "left", "top")
VECTORS = {"right": (1, 0), "bottom": (0, 1), "left": (-1, 0), "top": (0, -1)}
SIZES = {"Inport": (72, 34), "Outport": (72, 34), "Sum": (38, 38),
         "Integrator": (64, 66), "TransferFcn": (126, 76), "MatlabFunction": (170, 84),
         "Subsystem": (176, 88), "StateflowChart": (190, 100), "Constant": (66, 44), "Gain": (74, 58), "Mux": (30, 76)}


def _port_counts(graph):
    counts = defaultdict(lambda: [0, 0])
    for edge in graph["edges"]:
        for side, endpoint in ((0, edge["destination"]), (1, edge["source"])):
            if isinstance(endpoint["port"], int):
                counts[endpoint["block_id"]][side] = max(counts[endpoint["block_id"]][side], endpoint["port"] + 1)
    return counts


def _layers(graph):
    """Condense cycles before ranking; retain every node in feedback loops."""
    ids = [node["id"] for node in graph["nodes"]]
    adjacency = defaultdict(list)
    reverse = defaultdict(list)
    for edge in graph["edges"]:
        a, b = edge["source"]["block_id"], edge["destination"]["block_id"]
        adjacency[a].append(b)
        reverse[b].append(a)
    visited, order = set(), []
    for node in ids:
        stack = [(node, False)]
        while stack:
            current, finished = stack.pop()
            if finished:
                order.append(current)
            elif current not in visited:
                visited.add(current)
                stack.append((current, True))
                stack.extend((child, False) for child in reversed(adjacency[current]) if child not in visited)
    components, membership = [], {}
    for node in reversed(order):
        if node in membership:
            continue
        number = len(components)
        component, stack = [], [node]
        while stack:
            current = stack.pop()
            if current in membership:
                continue
            membership[current] = number
            component.append(current)
            stack.extend(reverse[current])
        components.append(component)
    outgoing, incoming = defaultdict(set), defaultdict(set)
    for a, children in adjacency.items():
        for b in children:
            if membership[a] != membership[b]:
                outgoing[membership[a]].add(membership[b])
                incoming[membership[b]].add(membership[a])
    degree = {i: len(incoming[i]) for i in range(len(components))}
    local = {}
    spans = {}
    by_id = {node["id"]: node for node in graph["nodes"]}
    for number, component in enumerate(components):
        degrees = {node: 0 for node in component}
        forward = defaultdict(set)
        for source in component:
            for target in adjacency[source]:
                if target in degrees and by_id[target]["type"] not in ("Integrator", "TransferFcn", "StateflowChart", "Subsystem"):
                    forward[source].add(target)
        for targets in forward.values():
            for target in targets:
                degrees[target] += 1
        ready = deque(sorted(node for node in component if not degrees[node]))
        ranks = {node: 0 for node in component}
        processed = set()
        while ready:
            source = ready.popleft(); processed.add(source)
            for target in sorted(forward[source]):
                ranks[target] = max(ranks[target], ranks[source] + 1)
                degrees[target] -= 1
                if not degrees[target]: ready.append(target)
        for node in sorted(set(component) - processed):
            ranks[node] = max(ranks.values(), default=-1) + 1
        local.update(ranks)
        spans[number] = max(ranks.values()) + 1
    rank = {i: 0 for i in degree}
    queue = deque(i for i in degree if not degree[i])
    while queue:
        current = queue.popleft()
        for child in sorted(outgoing[current]):
            rank[child] = max(rank[child], rank[current] + spans[current])
            degree[child] -= 1
            if not degree[child]:
                queue.append(child)
    return {node: rank[membership[node]] + local[node] for node in ids}


def _geometry(graph, hints, native):
    counts = _port_counts(graph)
    geometries = {}
    positions = hints.get("nodes", {})
    ranked = _layers(graph)
    if native:
        origin_x = min(value["bounds"][0] for value in positions.values())
        origin_y = min(value["bounds"][1] for value in positions.values())
    columns = defaultdict(list)
    for node in graph["nodes"]:
        hint = positions.get(node["id"], {})
        count = [max(counts[node["id"]][i], hint.get(key, 0)) for i, key in enumerate(("input_count", "output_count"))]
        width, height = SIZES.get(node["type"], (116, 66))
        if node["type"] == "Sum":
            width = height = max(width, max(count) * 12 + 12)
        else:
            height = max(height, max(count) * 19 + 16)
        label = node.get("display_name") or hint.get("display_name") or node["name"].split("/")[-1]
        label_width = min(180, max(width, len(label) * 7))
        rotation = hint.get("rotation", 0) if native else 0
        sides = {"input": ["left"] * count[0], "output": ["right"] * count[1]}
        signs = hint.get("sum_inputs", "")
        if native and node["type"] == "Sum" and len(signs) == 3 and "|" in signs:
            sides["input"] = [side for side, sign in zip(("top", "left", "bottom"), signs) if sign != "|"]
            sides["input"] += ["left"] * max(0, count[0] - len(sides["input"]))
        for direction in sides:
            sides[direction] = [SIDES[(SIDES.index(side) + rotation // 90) % 4] for side in sides[direction]]
            if native and hint.get("mirror"):
                sides[direction] = [{"left": "right", "right": "left"}.get(side, side) for side in sides[direction]]
        # Keep captions away from vertical signal ports.
        used_sides = set(sides["input"] + sides["output"])
        label_side = "right" if {"top", "bottom"} <= used_sides else "top" if "bottom" in used_sides else "bottom"
        if native:
            bounds = hint["bounds"]
            center_x = 90 + ((bounds[0] + bounds[2]) / 2 - origin_x) * 2.65
            center_y = 100 + ((bounds[1] + bounds[3]) / 2 - origin_y) * 2.65
            x, y = center_x - width / 2, center_y - height / 2
        else:
            columns[ranked[node["id"]]].append(node)
            x, y = 0, 0
        geometries[node["id"]] = {"x": x, "y": y, "width": width, "height": height,
                                    "label": label, "label_width": label_width, "label_side": label_side,
                                    "sides": sides, "rotation": rotation}
    if not native:
        cursor_x = 72
        for rank, nodes in sorted(columns.items()):
            nodes.sort(key=lambda node: ((positions.get(node["id"], {}).get("bounds") or [0, ranked[node["id"]], 0, 0])[1], node["name"]))
            width = max(geometries[node["id"]]["width"] for node in nodes)
            cursor_y = 88
            for node in nodes:
                geometry = geometries[node["id"]]
                geometry.update(x=cursor_x + (width - geometry["width"]) / 2, y=cursor_y)
                cursor_y += geometry["height"] + 76
            cursor_x += width + 130
    # Native layouts with crowded labels are expanded locally without dropping blocks.
    settled = []
    for node in sorted(graph["nodes"], key=lambda n: (geometries[n["id"]]["y"], geometries[n["id"]]["x"])):
        geometry = geometries[node["id"]]
        while any(_overlap(_bounds(geometry), _bounds(other), 8) for other in settled):
            geometry["y"] += 12
        settled.append(geometry)
    for geometry in geometries.values():
        geometry["ports"] = {direction: [_anchor(geometry, index, direction) for index in range(len(sides))]
                             for direction, sides in geometry["sides"].items()}
    return geometries


def _anchor(node, port, direction):
    side = node["sides"][direction][port]
    peers = [index for index, value in enumerate(node["sides"][direction]) if value == side]
    ratio = (peers.index(port) + 1) / (len(peers) + 1)
    x, y, width, height = (node[key] for key in ("x", "y", "width", "height"))
    point = {"left": (x, y + height * ratio), "right": (x + width, y + height * ratio),
             "top": (x + width * ratio, y), "bottom": (x + width * ratio, y + height)}[side]
    return {"x": point[0], "y": point[1], "side": side, "port": port}


def _bounds(node):
    x, y, width, height = (node[key] for key in ("x", "y", "width", "height"))
    left, top, right, bottom = x, y, x + width, y + height
    if node["label_side"] == "right":
        right += node["label_width"] + 12
    elif node["label_side"] == "top":
        top -= 24
        left, right = min(left, x + width / 2 - node["label_width"] / 2), max(right, x + width / 2 + node["label_width"] / 2)
    else:
        bottom += 28
        left, right = min(left, x + width / 2 - node["label_width"] / 2), max(right, x + width / 2 + node["label_width"] / 2)
    return (left, top, right, bottom)


def _overlap(a, b, padding=0):
    return a[0] < b[2] + padding and a[2] + padding > b[0] and a[1] < b[3] + padding and a[3] + padding > b[1]


def _blocked(a, b, rectangles):
    for left, top, right, bottom in rectangles:
        if a[0] == b[0] and left < a[0] < right and max(a[1], b[1]) > top and min(a[1], b[1]) < bottom:
            return True
        if a[1] == b[1] and top < a[1] < bottom and max(a[0], b[0]) > left and min(a[0], b[0]) < right:
            return True
    return False


def _simplify(points):
    result = []
    for point in points:
        if result and point == result[-1]:
            continue
        if len(result) > 1 and ((result[-2][0] == result[-1][0] == point[0]) or (result[-2][1] == result[-1][1] == point[1])):
            result[-1] = point
        else:
            result.append(point)
    return result


def _route(start, end, rectangles, preferences=()):
    """A* over rectangle boundaries; cost discourages needless bends."""
    xs = sorted({start[0], end[0], *(x for rectangle in rectangles for x in (rectangle[0] - 20, rectangle[2] + 20)),
                 *(point[0] for point in preferences)})
    ys = sorted({start[1], end[1], *(y for rectangle in rectangles for y in (rectangle[1] - 20, rectangle[3] + 20)),
                 *(point[1] for point in preferences)})
    xs = sorted({*xs, min(xs) - 40, max(xs) + 40})
    ys = sorted({*ys, min(ys) - 40, max(ys) + 40})
    start_index, end_index = (xs.index(start[0]), ys.index(start[1]), 0), (xs.index(end[0]), ys.index(end[1]))
    queue, costs, parents = [(0, 0, start_index)], {start_index: 0}, {}
    serial = 0
    goal = None
    # Bound work for large models; an unverified route is never rendered.
    while queue and serial < 150000:
        _, _, current = heapq.heappop(queue)
        ix, iy, previous_direction = current
        if (ix, iy) == end_index:
            goal = current
            break
        a = (xs[ix], ys[iy])
        for dx, dy, direction in ((-1, 0, 1), (1, 0, 1), (0, -1, 2), (0, 1, 2)):
            nx, ny = ix + dx, iy + dy
            if not (0 <= nx < len(xs) and 0 <= ny < len(ys)):
                continue
            b = (xs[nx], ys[ny])
            if _blocked(a, b, rectangles):
                continue
            state = (nx, ny, direction)
            distance = abs(a[0] - b[0]) + abs(a[1] - b[1])
            cost = costs[current] + distance + (24 if previous_direction and previous_direction != direction else 0)
            if cost >= costs.get(state, math.inf):
                continue
            costs[state], parents[state] = cost, current
            serial += 1
            heuristic = abs(b[0] - end[0]) + abs(b[1] - end[1])
            heapq.heappush(queue, (cost + heuristic, serial, state))
    if goal is None:
        raise ValueError("Could not find a verified wire route without crossing blocks; diagram was not generated.")
    path = []
    while goal in parents:
        path.append((xs[goal[0]], ys[goal[1]]))
        goal = parents[goal]
    path.append(start)
    return _simplify(list(reversed(path))), False


def _build(graph, hints, native):
    nodes = _geometry(graph, hints, native)
    rectangles = [_bounds(node) for node in nodes.values()]
    routes = []
    native_routes = {(item["source"]["block_id"], item["source"]["port"], item["destination"]["block_id"], item["destination"]["port"]): item for item in hints.get("routes", [])}
    for index, edge in enumerate(graph["edges"]):
        source = nodes[edge["source"]["block_id"]]
        destination = nodes[edge["destination"]["block_id"]]
        start = source["ports"]["output"][edge["source"]["port"]]
        if isinstance(edge["destination"]["port"], int):
            end = destination["ports"]["input"][edge["destination"]["port"]]
        else:
            end = {"x": destination["x"] + destination["width"] / 2, "y": destination["y"], "side": "top"}
        a, b = (start["x"], start["y"]), (end["x"], end["y"])
        def escape(point, side, node):
            left, top, right, bottom = _bounds(node)
            return {"left": (left - 24, point[1]), "right": (right + 24, point[1]),
                    "top": (point[0], top - 24), "bottom": (point[0], bottom + 24)}[side]
        escape_a, escape_b = escape(a, start["side"], source), escape(b, end["side"], destination)
        hint = native_routes.get((edge["source"]["block_id"], edge["source"]["port"], edge["destination"]["block_id"], edge["destination"]["port"]), {})
        preferences, point = [], escape_a
        if native:
            for dx, dy in hint.get("offsets", []):
                point = (point[0] + dx * 2.65, point[1] + dy * 2.65)
                preferences.append(point)
        path, fallback = _route(escape_a, escape_b, rectangles, preferences)
        points = _simplify([a, *path, b])
        routes.append({"index": index, "points": points, "label": hint.get("label", ""), "fallback": fallback})
    bounds = [_bounds(node) for node in nodes.values()]
    points = [point for route in routes for point in route["points"]]
    left = min([box[0] for box in bounds] + [p[0] for p in points]) - 42
    top = min([box[1] for box in bounds] + [p[1] for p in points]) - 42
    right = max([box[2] for box in bounds] + [p[0] for p in points]) + 42
    bottom = max([box[3] for box in bounds] + [p[1] for p in points]) + 42
    return {"nodes": nodes, "edges": routes, "bounds": [left, top, right - left, bottom - top]}


def diagram_layouts(graph, hints=None):
    hints = hints or {}
    native_nodes = hints.get("nodes", {})
    result = {"flow": _build(graph, hints, False)}
    if all(node["id"] in native_nodes and native_nodes[node["id"]].get("is_root") for node in graph["nodes"]):
        result["native"] = _build(graph, hints, True)
    return result
