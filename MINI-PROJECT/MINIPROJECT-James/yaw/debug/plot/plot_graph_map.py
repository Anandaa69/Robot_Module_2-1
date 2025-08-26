import json
import matplotlib.pyplot as plt

def plot_graph_from_file(filename="mapped_nodes.json", save_as=None):
    """
    พล็อตแผนที่จากไฟล์ JSON ที่เซฟจาก GraphMapper
    แสดงโหนด, เส้นทาง, และกำแพงรอบช่องตาม property 'walls'
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for node_id, node_data in data.items():
        x, y = node_data["position"]

        # วาดโหนดเป็นจุดสีน้ำเงิน
        ax.plot(x, y, "bo", markersize=6)
        ax.text(x + 0.1, y + 0.1, node_id, fontsize=7)

        # วาดกำแพงรอบ cell ตาม property 'walls'
        walls = node_data.get("walls", {})
        half = 0.5  # ครึ่งความยาวของ cell

        if walls.get("north", False):
            ax.plot([x - half, x + half], [y + half, y + half], "k-", linewidth=2)
        if walls.get("south", False):
            ax.plot([x - half, x + half], [y - half, y - half], "k-", linewidth=2)
        if walls.get("east", False):
            ax.plot([x + half, x + half], [y - half, y + half], "k-", linewidth=2)
        if walls.get("west", False):
            ax.plot([x - half, x - half], [y - half, y + half], "k-", linewidth=2)

        # วาดเส้นทางไปเพื่อนบ้าน (ทางออก)
        neighbors = node_data.get("neighbors", {})
        if neighbors:
            for neighbor in neighbors.values():
                if isinstance(neighbor, dict) and "position" in neighbor:
                    nx, ny = neighbor["position"]
                    ax.plot([x, nx], [y, ny], "g--", linewidth=1)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Robot Mapped Graph")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    if save_as:
        plt.savefig(save_as, dpi=300)
        print(f"✅ Map saved as {save_as}")
    else:
        plt.show()
