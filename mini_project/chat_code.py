"""
robomaster_s1_tof_mapping.py

Example pipeline for RoboMaster S1 using a single front ToF sensor + gimbal active scanning
to build an occupancy-grid map, detect markers, and explore via frontier-based exploration.

This is a runnable-ish template (not a drop-in production system). Replace the placeholder
RoboMaster SDK calls with your actual SDK commands and tune parameters for your robot.

Features:
- Occupancy grid (configurable resolution)
- Active scanning (rotate gimbal, sample ToF at increments)
- Ray-casting update of grid
- Simple marker detection (OpenCV ArUco)
- Frontier detection + selection (nearest frontier)
- A* path planning on grid
- Simple local following (step along path, re-scan at waypoints)

Assumptions:
- You have a camera for ArUco detection (frame given by get_camera_frame())
- You have a function get_tof_distance() that returns distance in meters from front ToF
- You can command robot to move relative distances or to set chassis wheel speeds
- Odometry is read via get_odometry() returning (x, y, yaw) in meters/radians

Tune constants at the top of the file.

"""

import time
import math
import numpy as np
import heapq
import cv2

# ----------------------- CONFIG -----------------------
MAP_SIZE_M = 4.2
RES = 0.03                 # meters per cell (30 mm)
GRID_W = int(MAP_SIZE_M / RES)
GRID_ORIGIN = (GRID_W // 2, GRID_W // 2)  # place robot start near center

TOF_MAX_RANGE = 2.0        # meters (cap for raycast)
SCAN_ANGLE_STEP = 8        # degrees between ToF samples during active scan
ROBOT_RADIUS_M = 0.12      # robot radius for inflation
INFLATION_CELLS = int(math.ceil(ROBOT_RADIUS_M / RES))

SCAN_SETTLE = 0.06         # seconds to wait after rotating before reading ToF
MOVE_STEP = 0.18           # meters per move step when following path
RESCAN_AT_NODE = True

# ArUco config
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()

# ------------------- OCCUPANCY GRID -------------------
# Grid values: -1 unknown, 0 free, 1 occupied
grid = np.full((GRID_W, GRID_W), -1, dtype=np.int8)
visited = np.zeros((GRID_W, GRID_W), dtype=bool)
marker_db = {}  # id -> (x, y, ts)

# ------------------- UTILITIES -------------------

def world_to_grid(x, y):
    # robot start at world (0,0) mapped to grid origin
    gx = int(round(GRID_ORIGIN[0] + x / RES))
    gy = int(round(GRID_ORIGIN[1] - y / RES))  # y axis inverted for display
    return gx, gy


def grid_to_world(gx, gy):
    x = (gx - GRID_ORIGIN[0]) * RES
    y = (GRID_ORIGIN[1] - gy) * RES
    return x, y


def in_bounds(gx, gy):
    return 0 <= gx < GRID_W and 0 <= gy < GRID_W

# Bresenham line (grid) for ray casting
def bresenham(x0, y0, x1, y1):
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def inflate_obstacles():
    # create inflated grid for planning
    inflated = np.copy(grid)
    occ = np.where(grid == 1)
    for (gx, gy) in zip(occ[0], occ[1]):
        for dx in range(-INFLATION_CELLS, INFLATION_CELLS + 1):
            for dy in range(-INFLATION_CELLS, INFLATION_CELLS + 1):
                nx, ny = gx + dx, gy + dy
                if in_bounds(nx, ny) and inflated[nx, ny] != 1:
                    inflated[nx, ny] = 1
    return inflated

# ------------------- PLACEHOLDER ROBOT I/O -------------------
# Replace these with real RoboMaster SDK calls

def init_robot():
    # placeholder
    print("[robot] init (replace with SDK init)")
    return None


def get_odometry():
    # return (x, y, yaw) in meters / radians
    # placeholder: should read from robot's odometry / chassis status
    return odom_state['x'], odom_state['y'], odom_state['yaw']


def set_drive_target_linear(dx, dy, speed=0.3):
    # move robot relatively (blocking or non-blocking depending on SDK)
    # you should implement motion controller or use SDK's move distance
    print(f"[robot] move Δ=({dx:.2f},{dy:.2f})")
    # naive: update odom_state for simulation
    odom_state['x'] += dx
    odom_state['y'] += dy


def rotate_gimbal_to(angle_deg):
    # send gimbal to absolute yaw in degrees (0 forward)
    # placeholder
    odom_state['gimbal_yaw'] = angle_deg
    time.sleep(0.01)


def get_tof_distance():
    # read ToF sensor (meters)
    # placeholder returns a fake distance; replace with real sensor read
    # For demo, return max range
    return TOF_MAX_RANGE


def get_camera_frame():
    # return an image frame from front camera (BGR numpy array)
    # placeholder: return None
    return None

# ------------------- MAPPING FUNCTIONS -------------------

def raycast_update(robot_x, robot_y, robot_yaw, angle_deg, dist_m):
    # angle_deg relative to robot forward, dist_m measured
    # Convert to world coords
    angle_rad = math.radians(angle_deg) + robot_yaw
    end_x = robot_x + dist_m * math.cos(angle_rad)
    end_y = robot_y + dist_m * math.sin(angle_rad)

    gx0, gy0 = world_to_grid(robot_x, robot_y)
    gx1, gy1 = world_to_grid(end_x, end_y)

    line = bresenham(gx0, gy0, gx1, gy1)
    # mark free until last cell; if dist < max_range then last cell is occupied
    last_idx = len(line) - 1
    if dist_m >= TOF_MAX_RANGE - 1e-3:
        # no hit: mark all as free
        for (cx, cy) in line:
            if in_bounds(cx, cy):
                grid[cx, cy] = 0
    else:
        for i, (cx, cy) in enumerate(line):
            if not in_bounds(cx, cy):
                break
            if i < last_idx:
                grid[cx, cy] = 0
            else:
                grid[cx, cy] = 1


def active_scan_and_update():
    # perform active scan by rotating gimbal and reading ToF at angular steps
    robot_x, robot_y, robot_yaw = get_odometry()
    # We'll scan relative angles -180..+180
    for a in range(-180, 180, SCAN_ANGLE_STEP):
        rotate_gimbal_to(a)
        time.sleep(SCAN_SETTLE)
        d = get_tof_distance()
        # ignore nan or faulty readings
        if d is None or math.isnan(d) or d <= 0:
            continue
        d = min(d, TOF_MAX_RANGE)
        raycast_update(robot_x, robot_y, robot_yaw, a, d)

# ------------------- ARUCO DETECTION -------------------

def detect_markers_and_correct_pose():
    frame = get_camera_frame()
    if frame is None:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is None:
        return
    robot_x, robot_y, robot_yaw = get_odometry()
    for i, c in enumerate(corners):
        mid = c[0].mean(axis=0)
        marker_id = int(ids[i])
        # Placeholder: compute marker relative pose using solvePnP if you have marker size & camera intrinsics
        # Here we'll simply record approximate location in world using current odometry + forward distance
        # In practice: use camera intrinsics + marker corners -> rvec,tvec -> transform to world
        approx_dist = 0.6  # placeholder estimate
        angle_offset = 0.0
        mx = robot_x + approx_dist * math.cos(robot_yaw + angle_offset)
        my = robot_y + approx_dist * math.sin(robot_yaw + angle_offset)
        marker_db[marker_id] = (mx, my, time.time())
        print(f"detected marker {marker_id} ~({mx:.2f},{my:.2f})")
        # Optionally: correct odometry here using marker location if you know marker true position

# ------------------- FRONTIER DETECTION -------------------

def find_frontiers():
    frontiers = []
    # a frontier cell: unknown (-1) and has a neighbor free (0)
    for gx in range(GRID_W):
        for gy in range(GRID_W):
            if grid[gx, gy] != -1:
                continue
            neighbors = [(gx + dx, gy + dy) for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]]
            for nx, ny in neighbors:
                if in_bounds(nx, ny) and grid[nx, ny] == 0:
                    frontiers.append((gx, gy))
                    break
    return frontiers

# choose nearest frontier by path length (A* cost) or euclidean
def select_frontier(frontiers, robot_pos):
    if not frontiers:
        return None
    rx, ry = robot_pos
    rgx, rgy = world_to_grid(rx, ry)
    best = None
    best_cost = float('inf')
    inflated = inflate_obstacles()
    for (fx, fy) in frontiers:
        # quick euclidean filter
        ex, ey = grid_to_world(fx, fy)
        d = math.hypot(ex - rx, ey - ry)
        if d > 6.0:
            continue
        path = a_star((rgx, rgy), (fx, fy), inflated)
        if path is None:
            continue
        cost = len(path)
        if cost < best_cost:
            best_cost = cost
            best = (fx, fy)
    return best

# ------------------- A* PLANNER -------------------

def a_star(start, goal, obstacle_grid):
    sx, sy = start; gx, gy = goal
    if not in_bounds(gx, gy) or not in_bounds(sx, sy):
        return None
    if obstacle_grid[gx, gy] == 1:
        return None

    open_set = []
    heapq.heappush(open_set, (0, (sx, sy)))
    came_from = {}
    gscore = { (sx, sy): 0 }

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (gx, gy):
            # reconstruct
            path = []
            cur = current
            while cur != (sx, sy):
                path.append(cur)
                cur = came_from[cur]
            path.append((sx, sy))
            path.reverse()
            return path
        cx, cy = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny):
                continue
            if obstacle_grid[nx, ny] == 1:
                continue
            tentative = gscore[current] + 1
            if (nx, ny) not in gscore or tentative < gscore[(nx, ny)]:
                gscore[(nx, ny)] = tentative
                f = tentative + h((nx, ny), (gx, gy))
                heapq.heappush(open_set, (f, (nx, ny)))
                came_from[(nx, ny)] = current
    return None

# ------------------- PATH FOLLOWING -------------------

def follow_path_world(path_cells):
    # path_cells: list of grid cells (gx, gy). We'll step along in world coordinates with MOVE_STEP
    for (gx, gy) in path_cells:
        wx, wy = grid_to_world(gx, gy)
        rx, ry, ryaw = get_odometry()
        dx = wx - rx
        dy = wy - ry
        dist = math.hypot(dx, dy)
        if dist < 0.06:
            continue
        # move in small steps toward target
        steps = max(1, int(math.ceil(dist / MOVE_STEP)))
        for s in range(steps):
            step_dist = min(MOVE_STEP, dist - s * MOVE_STEP)
            if step_dist <= 0:
                break
            # compute delta in robot frame approximatively
            theta = math.atan2(dy, dx) - ryaw
            rel_dx = step_dist * math.cos(ryaw + theta)
            rel_dy = step_dist * math.sin(ryaw + theta)
            set_drive_target_linear(rel_dx, rel_dy)
            time.sleep(0.1)
        if RESCAN_AT_NODE:
            active_scan_and_update()
            detect_markers_and_correct_pose()

# ------------------- MAIN LOOP -------------------

def exploration_main_loop():
    init_robot()
    # initial active scan
    active_scan_and_update()
    detect_markers_and_correct_pose()

    while True:
        frontiers = find_frontiers()
        print(f"frontiers: {len(frontiers)}")
        if not frontiers:
            print("Exploration complete")
            break
        rx, ry, ryaw = get_odometry()
        goal_cell = select_frontier(frontiers, (rx, ry))
        if goal_cell is None:
            print("No reachable frontier found. Exploration halted.")
            break
        sx, sy = world_to_grid(rx, ry)
        inflated = inflate_obstacles()
        path = a_star((sx, sy), goal_cell, inflated)
        if path is None:
            # mark frontier unreachable and continue
            gx, gy = goal_cell
            grid[gx, gy] = 1  # mark as occupied to avoid retrying
            continue
        # convert path (grid cells) to world and follow
        follow_path_world(path)
        # after reaching, do a full scan
        active_scan_and_update()
        detect_markers_and_correct_pose()

# ------------------- SIMULATION / DEMO STUB -------------------
# Minimal odom_state for demo; replace with real odometry system
odom_state = {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'gimbal_yaw': 0}

if __name__ == '__main__':
    try:
        exploration_main_loop()
    except KeyboardInterrupt:
        print("Stopped by user")

# ------------------- NOTES -------------------
# - Replace placeholders with real sensor calls and robot motion commands.
# - For marker-based accurate pose correction, implement solvePnP with known marker size
#   and camera intrinsics, then transform marker pose into world coordinates.
# - If odometry drift is severe, consider running a particle filter or EKF.
# - Tune RES, SCAN_ANGLE_STEP, MOVE_STEP for best performance in the real maze.
# - Consider saving map to file at the end (numpy.save) and visualizing with matplotlib.
