# plot_maze_explored.py
# วาดแผนที่สำรวจ (ภาพนิ่ง + อนิเมชันเดินทีละช่อง ไม่ข้ามกำแพง)
# อ่านไฟล์: Assignment\data\maze_data_true_final.json

from pathlib import Path
import json
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ====================== CONFIG ======================
JSON_PATH = Path("Assignment") / "data" / "maze_data_2.json"

# แสดง/บันทึกผล
SAVE_STATIC = True                  # เซฟภาพนิ่ง PNG
STATIC_OUT = Path("maze_static.png")

SAVE_ANIM = False                   # เซฟอนิเมชัน (ตั้ง True เมื่อต้องการ)
# หมายเหตุ: .mp4 ต้องมี ffmpeg, .gif ต้องติดตั้ง pillow
ANIM_OUT = Path("maze_animation.gif")
ANIM_INTERVAL_MS = 220             # เร็ว/ช้าของการเดิน (ms ต่อก้าว)
ANIM_REPEAT = True
ANIM_CACHE_FRAMES = False          # ลดเมม/แก้บั๊กบาง backend

# โหมดเริ่ม BFS ที่ไหน
START_KEY = None                   # None = เลือกอัตโนมัติ ("0_0" ถ้ามี)

# เลเยอร์แสดงผล
SHOW_ALL_NODES_FAINT = True
FAINT_ALPHA = 0.25

# สี/ขนาด
COLOR_NODE = "#1f77b4"
COLOR_DEADEND = "#d62728"
COLOR_WALL = "black"
COLOR_HILIGHT = "#ff7f0e"
COLOR_TRAIL = "#2ca02c"
NODE_SIZE = 60
NODE_SIZE_DE = 90
# ====================================================

DIR_VEC = {
    "north": (0, 1),
    "south": (0, -1),
    "east":  (1, 0),
    "west":  (-1, 0),
}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}


def load_maze(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    raw_nodes = data.get("nodes", {})

    nodes = {}
    for key, nd in raw_nodes.items():
        pos = nd.get("position")
        if pos is None:
            # รองรับคีย์แบบ "x_y"
            try:
                x_str, y_str = key.split("_")
                pos = [int(x_str), int(y_str)]
            except Exception:
                continue
        nodes[key] = {
            "position": (int(pos[0]), int(pos[1])),
            "walls": dict(nd.get("walls", {})),
            "is_dead_end": bool(nd.get("is_dead_end", False)),
            "explored_directions": list(nd.get("explored_directions", [])),
            "unexplored_exits": list(nd.get("unexplored_exits", [])),
        }

    bounds = meta.get("boundaries")
    if not bounds:
        xs = [nd["position"][0] for nd in nodes.values()]
        ys = [nd["position"][1] for nd in nodes.values()]
        bounds = {
            "min_x": min(xs) - 1,
            "max_x": max(xs) + 1,
            "min_y": min(ys) - 1,
            "max_y": max(ys) + 1,
        }

    return meta, nodes, bounds


def wall_segments(nodes):
    """แปลง walls ของแต่ละเซลล์เป็นเส้น (x1, y1, x2, y2) แบบไม่ซ้ำ"""
    segs = set()

    def add(a, b):
        if a > b:
            a, b = b, a
        segs.add((a[0], a[1], b[0], b[1]))

    for nd in nodes.values():
        x, y = nd["position"]
        cx, cy = float(x), float(y)
        half = 0.5
        w = nd["walls"]

        if w.get("north"):
            add((cx - half, cy + half), (cx + half, cy + half))
        if w.get("south"):
            add((cx - half, cy - half), (cx + half, cy - half))
        if w.get("east"):
            add((cx + half, cy - half), (cx + half, cy + half))
        if w.get("west"):
            add((cx - half, cy - half), (cx - half, cy + half))

    return list(segs)


def build_graph(nodes):
    """สร้างกราฟการเชื่อมจากกำแพง (ไม่มีผนังสองฝั่ง -> เดินได้)"""
    pos_to_key = {nd["position"]: k for k, nd in nodes.items()}
    adj = defaultdict(list)

    for k, nd in nodes.items():
        x, y = nd["position"]
        walls = nd["walls"]
        for d, (dx, dy) in DIR_VEC.items():
            if not walls.get(d, False):
                nb_pos = (x + dx, y + dy)
                nb_key = pos_to_key.get(nb_pos)
                if nb_key:
                    nb_walls = nodes[nb_key]["walls"]
                    if not nb_walls.get(OPPOSITE[d], False):
                        adj[k].append(nb_key)
    return adj


def bfs_visit_order(nodes, start_key=None):
    """คืนลำดับ 'การค้นพบ' แบบ BFS (ใช้ทำลิสต์จุดที่อยากไปเยี่ยมชม)"""
    if not nodes:
        return []

    if start_key is None:
        start_key = "0_0" if "0_0" in nodes else sorted(
            nodes.keys(),
            key=lambda k: (-nodes[k]["position"][1], nodes[k]["position"][0])
        )[0]

    adj = build_graph(nodes)
    visited = {start_key}
    q = deque([start_key])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)

    # ต่อโหนดโดดเดี่ยวไว้ท้าย
    if len(order) < len(nodes):
        for k in nodes.keys():
            if k not in visited:
                order.append(k)
    return order


def shortest_path_nodes(nodes, src_key, dst_key, adj=None):
    """หา shortest path (เป็นลิสต์คีย์โหนด) จาก src ไป dst บนกราฟ adj"""
    if src_key == dst_key:
        return [src_key]
    if adj is None:
        adj = build_graph(nodes)
    q = deque([src_key])
    parent = {src_key: None}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                if v == dst_key:
                    # สร้างเส้นทางคืน
                    path = [v]
                    while u is not None:
                        path.append(u)
                        u = parent[u]
                    path.reverse()
                    return path
                q.append(v)
    # ไม่เจอเส้นทาง (คนละคอมโพเนนต์) -> คืน dst เดี่ยวๆ เพื่อ "วาร์ป" อย่างสุภาพ
    return [src_key, dst_key]


def expand_visit_to_walk(nodes, visit_order):
    """
    แปลงลำดับเยี่ยมชม (ค้นพบ) ให้เป็น 'เส้นทางเดินจริง' ทีละช่อง:
    เชื่อมระหว่างจุดติดกันใน visit_order ด้วย shortest path แล้วรวมเป็นลิสต์ใหญ่
    """
    if not visit_order:
        return []
    adj = build_graph(nodes)
    walk = [visit_order[0]]
    for i in range(len(visit_order) - 1):
        a, b = visit_order[i], visit_order[i + 1]
        seg = shortest_path_nodes(nodes, a, b, adj=adj)
        # ต่อท่อ: ข้ามซ้ำตัวแรก
        if len(seg) > 1:
            walk.extend(seg[1:])
        else:
            walk.extend(seg)
    return walk


def setup_axes(bounds, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    min_x, max_x = bounds["min_x"], bounds["max_x"]
    min_y, max_y = bounds["min_y"], bounds["max_y"]

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x - 0.6, max_x + 0.6)
    ax.set_ylim(min_y - 0.6, max_y + 0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.5, alpha=0.25)
    if title:
        ax.set_title(title)

    # กรอบขอบเขต
    rect_x = [min_x - 0.5, max_x + 0.5, max_x + 0.5, min_x - 0.5, min_x - 0.5]
    rect_y = [min_y - 0.5, min_y - 0.5, max_y + 0.5, max_y + 0.5, min_y - 0.5]
    ax.plot(rect_x, rect_y, linewidth=1.2, alpha=0.6)
    return fig, ax


def draw_static(nodes, bounds, meta=None):
    fig, ax = setup_axes(bounds, title="Explored Maze (Static)")

    # ผนัง
    for (x1, y1, x2, y2) in wall_segments(nodes):
        ax.plot([x1, x2], [y1, y2], color=COLOR_WALL, linewidth=2)

    # โหนด
    xs, ys, sizes, cols = [], [], [], []
    for nd in nodes.values():
        x, y = nd["position"]
        xs.append(x)
        ys.append(y)
        if nd["is_dead_end"]:
            sizes.append(NODE_SIZE_DE)
            cols.append(COLOR_DEADEND)
        else:
            sizes.append(NODE_SIZE)
            cols.append(COLOR_NODE)

    ax.scatter(xs, ys, s=sizes, c=cols, edgecolor="white", linewidth=0.7, zorder=3)

    # ป้ายพิกัด
    for nd in nodes.values():
        x, y = nd["position"]
        ax.text(x, y + 0.18, f"{x},{y}", ha="center", va="bottom", fontsize=8, alpha=0.9)

    # กล่องข้อมูล
    if meta:
        info = [
            f"nodes: {len(nodes)}",
            f"drift corrections: {meta.get('drift_corrections', 0)}",
        ]
        ax.text(
            0.01, 0.99, "\n".join(info), transform=ax.transAxes,
            ha="left", va="top", fontsize=8, alpha=0.85,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8")
        )

    plt.tight_layout()
    if SAVE_STATIC:
        fig.savefig(STATIC_OUT, dpi=200)
        print(f"[Saved] {STATIC_OUT}")
    return fig, ax


def animate_walk(nodes, bounds, meta=None, start_key=None):
    # 1) ลำดับเยี่ยมชม (ค้นพบ)
    visit_order = bfs_visit_order(nodes, start_key=start_key)
    if not visit_order:
        fig, ax = setup_axes(bounds, title="Exploration Animation (no nodes)")
        ax.text(0.5, 0.5, "No nodes to animate", transform=ax.transAxes,
                ha="center", va="center", fontsize=12)
        return fig, None

    # 2) แปลงเป็น 'ทางเดินจริง' ทีละช่อง (ไม่มีการข้ามกำแพง)
    walk_keys = expand_visit_to_walk(nodes, visit_order)

    # เตรียมข้อมูลพิกัด
    pos = {k: nodes[k]["position"] for k in nodes.keys()}
    walk_xy = np.array([pos[k] for k in walk_keys], dtype=float)

    # === วาดพื้น ===
    fig, ax = setup_axes(bounds, title="Exploration Animation (step-by-step walk)")

    # ผนังคงที่
    for (x1, y1, x2, y2) in wall_segments(nodes):
        ax.plot([x1, x2], [y1, y2], color=COLOR_WALL, linewidth=2, alpha=0.95)

    # ทุกโหนดแบบจาง
    if SHOW_ALL_NODES_FAINT:
        xs_all = [nd["position"][0] for nd in nodes.values()]
        ys_all = [nd["position"][1] for nd in nodes.values()]
        ax.scatter(xs_all, ys_all, s=30, c="gray", alpha=FAINT_ALPHA, zorder=1)

    # trail (เส้นทางที่เดินแล้ว)
    trail_line, = ax.plot([], [], linewidth=2.5, color=COLOR_TRAIL, alpha=0.9, zorder=2)

    # จุดที่เยือนแล้ว (สีจริง)
    visited_scat = ax.scatter([], [], s=[], c=[], edgecolor="white", linewidth=0.7, zorder=3)

    # ไฮไลต์ตำแหน่งปัจจุบัน
    (highlight,) = ax.plot([], [], marker="o", markersize=14, markerfacecolor="none",
                           markeredgecolor=COLOR_HILIGHT, markeredgewidth=2, linestyle="", zorder=4)

    label_text = ax.text(0.99, 0.99, "", transform=ax.transAxes,
                         ha="right", va="top", fontsize=9, alpha=0.9,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"))

    # เตรียมโหนดสำหรับสี/ขนาดเมื่อถูกเยือน
    is_dead = {k: nodes[k]["is_dead_end"] for k in nodes.keys()}

    def init():
        trail_line.set_data([], [])
        visited_scat.set_offsets(np.empty((0, 2)))
        visited_scat.set_sizes(np.array([], dtype=float))
        visited_scat.set_color([])
        highlight.set_data([], [])
        label_text.set_text("")
        return trail_line, visited_scat, highlight, label_text

    def update(frame):
        # frame คือ index ใน walk_keys (เดินทีละช่อง)
        xs_trail = walk_xy[:frame + 1, 0]
        ys_trail = walk_xy[:frame + 1, 1]
        trail_line.set_data(xs_trail, ys_trail)

        # จุดที่เยือนแล้ว (unique ตามคีย์ เพื่อไม่ซ้ำเมื่อเดินทับ)
        visited_keys = list(dict.fromkeys(walk_keys[:frame + 1]))
        visited_xy = np.array([pos[k] for k in visited_keys], dtype=float)
        sizes = [NODE_SIZE_DE if is_dead[k] else NODE_SIZE for k in visited_keys]
        cols = [COLOR_DEADEND if is_dead[k] else COLOR_NODE for k in visited_keys]
        visited_scat.set_offsets(visited_xy if len(visited_xy) else np.empty((0, 2)))
        visited_scat.set_sizes(np.array(sizes, dtype=float))
        visited_scat.set_color(cols)

        # ไฮไลต์ปัจจุบัน
        cx, cy = walk_xy[frame]
        highlight.set_data([cx], [cy])

        # ข้อความ
        label_text.set_text(
            f"Step: {frame+1}/{len(walk_keys)}\n"
            f"Visit nodes: {len(visited_keys)}/{len(nodes)}"
        )

        return trail_line, visited_scat, highlight, label_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(walk_keys),
        init_func=init,
        interval=ANIM_INTERVAL_MS,
        blit=False,
        repeat=ANIM_REPEAT,
        cache_frame_data=ANIM_CACHE_FRAMES,
    )

    # บันทึกถ้าต้องการ
    if SAVE_ANIM:
        try:
            if ANIM_OUT.suffix.lower() == ".gif":
                anim.save(ANIM_OUT, dpi=160, writer="pillow")
            else:
                anim.save(ANIM_OUT, dpi=160)  # ต้องมี ffmpeg สำหรับ mp4
            print(f"[Saved] {ANIM_OUT}")
        except Exception as e:
            print("[Warn] Save animation failed:", e)

    plt.tight_layout()
    return fig, anim


def main():
    meta, nodes, bounds = load_maze(JSON_PATH)

    # ภาพนิ่ง
    draw_static(nodes, bounds, meta=meta)

    # อนิเมชัน: เดินทีละช่องตาม shortest path ระหว่างโหนดที่อยากเยี่ยมชม
    fig_anim, anim = animate_walk(nodes, bounds, meta=meta, start_key=START_KEY)

    # กัน GC ของแอนิเมชัน
    globals()["_ANIM_REF"] = anim

    plt.show()


if __name__ == "__main__":
    main()
