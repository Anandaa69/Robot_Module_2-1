# plot_maze_explored.py
# Matplotlib: วาดแผนที่สำรวจ (เวอร์ชันภาพนิ่ง + อนิเมชัน)
# อ่านจาก Assignment\data\maze_data_true_final.json

import json
import math
from collections import deque, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Config ----------
JSON_PATH = Path("Assignment") / "data" / "maze_data_true_final.json"
SAVE_STATIC = True          # เซฟภาพนิ่งเป็น PNG
STATIC_OUT = Path("maze_static.png")
SAVE_ANIM = False           # ถ้าต้องการเซฟอนิเมชันเป็น MP4 ให้ตั้ง True
ANIM_OUT = Path("maze_animation.mp4")
ANIM_INTERVAL_MS = 350      # ความเร็วอนิเมชันต่อเฟรม (ms)

# ---------- Helpers ----------

def load_maze(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    nodes = data.get("nodes", {})

    # ปรับตำแหน่งให้อยู่ใน list/tuple และ normalize คีย์
    norm_nodes = {}
    for key, nd in nodes.items():
        pos = nd.get("position", None)
        if pos is None:
            # key อาจเป็นรูป "x_y"
            try:
                x_str, y_str = key.split("_")
                pos = [int(x_str), int(y_str)]
            except Exception:
                continue
        nd = {
            "position": tuple(pos),
            "walls": nd.get("walls", {}),
            "is_dead_end": bool(nd.get("is_dead_end", False)),
            "explored_directions": list(nd.get("explored_directions", [])),
            "unexplored_exits": list(nd.get("unexplored_exits", [])),
        }
        norm_nodes[key] = nd

    # ขอบเขต: ใช้จาก metadata ถ้ามี ไม่งั้นคำนวณจากโหนด
    bounds = meta.get("boundaries", None)
    if not bounds:
        xs = [nd["position"][0] for nd in norm_nodes.values()]
        ys = [nd["position"][1] for nd in norm_nodes.values()]
        bounds = {
            "min_x": min(xs) - 1,
            "max_x": max(xs) + 1,
            "min_y": min(ys) - 1,
            "max_y": max(ys) + 1,
        }

    return meta, norm_nodes, bounds

def wall_segments(nodes):
    """
    แปลง walls ของแต่ละ cell เป็นเส้น (x1,y1,x2,y2) ไม่ซ้ำกัน
    สมมติ cell กว้าง 1 หน่วย วาง cell center ที่ (x, y)
    เส้นขอบ cell ใช้ครึ่งหน่วยจาก center
    """
    segs = set()
    # ใช้ lookup สำหรับ dedupe: เส้นเป็น tuple จัดอันดับจุดซ้ายก่อน
    def add_seg(a, b):
        # เรียงปลายทางให้ canonical
        if a > b:
            a, b = b, a
        segs.add((a[0], a[1], b[0], b[1]))

    for nd in nodes.values():
        x, y = nd["position"]
        cx, cy = float(x), float(y)
        half = 0.5

        w = nd["walls"]
        # เหนือ
        if w.get("north", False):
            a = (cx - half, cy + half)
            b = (cx + half, cy + half)
            add_seg(a, b)
        # ใต้
        if w.get("south", False):
            a = (cx - half, cy - half)
            b = (cx + half, cy - half)
            add_seg(a, b)
        # ตะวันออก
        if w.get("east", False):
            a = (cx + half, cy - half)
            b = (cx + half, cy + half)
            add_seg(a, b)
        # ตะวันตก
        if w.get("west", False):
            a = (cx - half, cy - half)
            b = (cx - half, cy + half)
            add_seg(a, b)

    return list(segs)

DIR_VEC = {
    "north": (0, 1),
    "south": (0, -1),
    "east":  (1, 0),
    "west":  (-1, 0),
}

OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}

def build_graph(nodes):
    """
    สร้างกราฟการเชื่อมต่อระหว่าง cell จากกำแพง: ถ้าไม่มีผนังในทิศ d และเพื่อนบ้านมีอยู่ → edge
    """
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
                    # ตรวจฝั่งตรงข้ามด้วย (กันกรณีข้อมูลผนังขัดแย้ง)
                    nb_walls = nodes[nb_key]["walls"]
                    if not nb_walls.get(OPPOSITE[d], False):
                        adj[k].append(nb_key)
    return adj

def bfs_order(nodes, start_key=None):
    """
    สร้างลำดับการเยี่ยมชมโหนดด้วย BFS เพื่อใช้อธิบายการ animate
    ถ้าไม่มี start_key จะพยายามใช้ "0_0" หรือคีย์แรกตามลำดับ
    """
    if not nodes:
        return []

    if start_key is None:
        if "0_0" in nodes:
            start_key = "0_0"
        else:
            # เลือกโหนดที่ y สูงสุด แล้ว x ต่ำสุด เพื่อให้เริ่มบนซ้าย (ปรับได้ตามใจ)
            start_key = sorted(nodes.keys(),
                               key=lambda k: (-nodes[k]["position"][1], nodes[k]["position"][0]))[0]

    adj = build_graph(nodes)
    visited = set([start_key])
    q = deque([start_key])
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                q.append(v)

    # ถ้าเป็นกราฟไม่เชื่อมทั้งหมด ให้ต่อโหนดที่เหลือเข้าไปตามลำดับ
    if len(order) < len(nodes):
        for k in nodes.keys():
            if k not in visited:
                order.append(k)

    return order

# ---------- Drawing ----------

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

    # วาดกรอบขอบเขต
    rect_x = [min_x - 0.5, max_x + 0.5, max_x + 0.5, min_x - 0.5, min_x - 0.5]
    rect_y = [min_y - 0.5, min_y - 0.5, max_y + 0.5, max_y + 0.5, min_y - 0.5]
    ax.plot(rect_x, rect_y, linewidth=1.2, alpha=0.6)

    return fig, ax

def draw_static(nodes, bounds, meta=None):
    fig, ax = setup_axes(bounds, title="Explored Maze (Static)")

    # วาดผนัง
    for (x1, y1, x2, y2) in wall_segments(nodes):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)

    # วาดโหนด
    xs = []
    ys = []
    colors = []
    sizes = []
    for nd in nodes.values():
        x, y = nd["position"]
        xs.append(x)
        ys.append(y)
        if nd["is_dead_end"]:
            colors.append("#d62728")   # แดง: ทางตัน
            sizes.append(90)
        else:
            colors.append("#1f77b4")   # น้ำเงิน: โหนดทั่วไป
            sizes.append(60)

    ax.scatter(xs, ys, s=sizes, c=colors, edgecolor="white", linewidth=0.7, zorder=3)

    # ติดป้ายเล็ก ๆ
    for nd in nodes.values():
        x, y = nd["position"]
        ax.text(x, y + 0.18, f"{x},{y}", ha="center", va="bottom", fontsize=8, alpha=0.9)

    # บอกจำนวน/เมตา
    if meta:
        info = [
            f"nodes: {len(nodes)}",
            f"drift corrections: {meta.get('drift_corrections', 0)}",
        ]
        ax.text(0.01, 0.99, "\n".join(info), transform=ax.transAxes,
                ha="left", va="top", fontsize=8, alpha=0.8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"))

    plt.tight_layout()
    if SAVE_STATIC:
        fig.savefig(STATIC_OUT, dpi=200)
        print(f"[Saved] {STATIC_OUT}")
    return fig, ax

def animate_exploration(nodes, bounds, meta=None, start_key=None):
    order = bfs_order(nodes, start_key=start_key)
    order_index = {k: i for i, k in enumerate(order)}

    fig, ax = setup_axes(bounds, title="Exploration Animation (BFS order)")
    # วาดผนังคงที่
    for (x1, y1, x2, y2) in wall_segments(nodes):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2, alpha=0.9)

    scat = ax.scatter([], [], s=[], c=[], edgecolor="white", linewidth=0.7, zorder=3)
    highlight, = ax.plot([], [], marker="o", markersize=14, markerfacecolor="none",
                         markeredgecolor="#ff7f0e", markeredgewidth=2, linestyle="")

    label_text = ax.text(0.99, 0.99, "", transform=ax.transAxes,
                         ha="right", va="top", fontsize=9, alpha=0.9,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8"))

    # เตรียมข้อมูลโหนดเรียงตามเฟรม
    node_keys = order
    node_data = [nodes[k] for k in node_keys]

    def init():
        scat.set_offsets([])
        scat.set_sizes([])
        scat.set_array(None)
        highlight.set_data([], [])
        label_text.set_text("")
        return scat, highlight, label_text

    def update(frame):
        # แสดงโหนดตั้งแต่ 0..frame
        xs, ys, sizes, cols = [], [], [], []
        for i in range(frame + 1):
            nd = node_data[i]
            x, y = nd["position"]
            xs.append(x)
            ys.append(y)
            sizes.append(90 if nd["is_dead_end"] else 60)
            cols.append("#d62728" if nd["is_dead_end"] else "#1f77b4")

        offsets = list(zip(xs, ys))
        scat.set_offsets(offsets)
        scat.set_sizes(sizes)
        scat.set_color(cols)

        # ไฮไลต์โหนดปัจจุบัน
        cx, cy = node_data[frame]["position"]
        highlight.set_data([cx], [cy])

        label_text.set_text(f"Step: {frame+1}/{len(node_data)}\nCurrent: {cx},{cy}")
        return scat, highlight, label_text

    anim = FuncAnimation(
        fig, update, frames=len(node_data), init_func=init,
        interval=ANIM_INTERVAL_MS, blit=True, repeat=True
    )

    # บันทึกไฟล์วิดีโอ (เลือกได้)
    if SAVE_ANIM:
        try:
            anim.save(ANIM_OUT, dpi=160)
            print(f"[Saved] {ANIM_OUT}")
        except Exception as e:
            print("[Warn] Save animation failed:", e)

    plt.tight_layout()
    return fig, anim

# ---------- Main ----------

def main():
    meta, nodes, bounds = load_maze(JSON_PATH)

    # 1) ภาพนิ่ง
    draw_static(nodes, bounds, meta=meta)

    # 2) อนิเมชัน (กดปิดหน้าต่างแรก จะเปิดหน้าต่างอนิเมชันต่อ)
    fig2, anim = animate_exploration(nodes, bounds, meta=meta, start_key=None)

    plt.show()

if __name__ == "__main__":
    main()
