import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy
from scipy.spatial.transform import Rotation as Rot

from queue import PriorityQueue
import sys
import pathlib
#sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from math import sin, cos, atan2, sqrt, acos, pi, hypot
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import Axes3D

# download the image files for mapping problem
import gdown

def rot_mat_2d(angle):
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]


def angle_mod(x, zero_2_2pi=False, degree=False):

    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

def plot_arrow(x, y, yaw, arrow_length=1.0,
               origin_point_plot_style="xr",
               head_width=0.1, fc="r", ec="k", **kwargs):
    if not isinstance(x, float):
        for (i_x, i_y, i_yaw) in zip(x, y, yaw):
            plot_arrow(i_x, i_y, i_yaw, head_width=head_width,
                       fc=fc, ec=ec, **kwargs)
    else:
        plt.arrow(x, y,
                  arrow_length * math.cos(yaw),
                  arrow_length * math.sin(yaw),
                  head_width=head_width,
                  fc=fc, ec=ec,
                  **kwargs)
        if origin_point_plot_style is not None:
            plt.plot(x, y, origin_point_plot_style)


def plot_curvature(x_list, y_list, heading_list, curvature,
                   k=0.01, c="-c", label="Curvature"):
    cx = [x + d * k * np.cos(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]
    cy = [y + d * k * np.sin(yaw - np.pi / 2.0) for x, y, yaw, d in
          zip(x_list, y_list, heading_list, curvature)]

    plt.plot(cx, cy, c, label=label)
    for ix, iy, icx, icy in zip(x_list, y_list, cx, cy):
        plt.plot([ix, icx], [iy, icy], c)


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


def plot_3d_vector_arrow(ax, p1, p2):
    setattr(Axes3D, 'arrow3D', _arrow3D)
    ax.arrow3D(p1[0], p1[1], p1[2],
               p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2],
               mutation_scale=20,
               arrowstyle="-|>",
               )



def plot_triangle(p1, p2, p3, ax):
    ax.add_collection3d(art3d.Poly3DCollection([[p1, p2, p3]], color='b'))


def set_equal_3d_axis(ax, x_lims, y_lims, z_lims):
    x_lims = np.asarray(x_lims)
    y_lims = np.asarray(y_lims)
    z_lims = np.asarray(z_lims)
    # compute max required range
    max_range = np.array([x_lims.max() - x_lims.min(),
                          y_lims.max() - y_lims.min(),
                          z_lims.max() - z_lims.min()]).max() / 2.0
    # compute mid-point along each axis
    mid_x = (x_lims.max() + x_lims.min()) * 0.5
    mid_y = (y_lims.max() + y_lims.min()) * 0.5
    mid_z = (z_lims.max() + z_lims.min()) * 0.5

    # set limits to axis
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def plan_dubins_path(s_x, s_y, s_yaw, g_x, g_y, g_yaw, curvature,
                     step_size=0.1, selected_types=None):
    if selected_types is None:
        planning_funcs = _PATH_TYPE_MAP.values()
    else:
        planning_funcs = [_PATH_TYPE_MAP[ptype] for ptype in selected_types]

    # calculate local goal x, y, yaw
    l_rot = rot_mat_2d(s_yaw)
    le_xy = np.stack([g_x - s_x, g_y - s_y]).T @ l_rot
    local_goal_x = le_xy[0]
    local_goal_y = le_xy[1]
    local_goal_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, modes, lengths, cost = _dubins_path_planning_from_origin(
        local_goal_x, local_goal_y, local_goal_yaw, curvature, step_size,
        planning_funcs)

    # Convert a local coordinate path to the global coordinate
    rot = rot_mat_2d(-s_yaw)
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = angle_mod(np.array(lp_yaw) + s_yaw)

    return x_list, y_list, yaw_list, modes, lengths, cost



def _mod2pi(theta):
    return angle_mod(theta, zero_2_2pi=True)


def _calc_trig_funcs(alpha, beta):
    sin_a = sin(alpha)
    sin_b = sin(beta)
    cos_a = cos(alpha)
    cos_b = cos(beta)
    cos_ab = cos(alpha - beta)
    return sin_a, sin_b, cos_a, cos_b, cos_ab


def _LSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "S", "L"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_a - sin_b))
    if p_squared < 0:  # invalid configuration
        return None, None, None, mode
    tmp = atan2((cos_b - cos_a), d + sin_a - sin_b)
    d1 = _mod2pi(-alpha + tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(beta - tmp)
    return d1, d2, d3, mode


def _RSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "S", "R"]
    p_squared = 2 + d ** 2 - (2 * cos_ab) + (2 * d * (sin_b - sin_a))
    if p_squared < 0:
        return None, None, None, mode
    tmp = atan2((cos_a - cos_b), d - sin_a + sin_b)
    d1 = _mod2pi(alpha - tmp)
    d2 = sqrt(p_squared)
    d3 = _mod2pi(-beta + tmp)
    return d1, d2, d3, mode


def _LSR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = -2 + d ** 2 + (2 * cos_ab) + (2 * d * (sin_a + sin_b))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((-cos_a - cos_b), (d + sin_a + sin_b)) - atan2(-2.0, d1)
    d2 = _mod2pi(-alpha + tmp)
    d3 = _mod2pi(-_mod2pi(beta) + tmp)
    return d2, d1, d3, mode


def _RSL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    p_squared = d ** 2 - 2 + (2 * cos_ab) - (2 * d * (sin_a + sin_b))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    d1 = sqrt(p_squared)
    tmp = atan2((cos_a + cos_b), (d - sin_a - sin_b)) - atan2(2.0, d1)
    d2 = _mod2pi(alpha - tmp)
    d3 = _mod2pi(beta - tmp)
    return d2, d1, d3, mode


def _RLR(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["R", "L", "R"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(alpha - atan2(cos_a - cos_b, d - sin_a + sin_b) + d2 / 2.0)
    d3 = _mod2pi(alpha - beta - d1 + d2)
    return d1, d2, d3, mode


def _LRL(alpha, beta, d):
    sin_a, sin_b, cos_a, cos_b, cos_ab = _calc_trig_funcs(alpha, beta)
    mode = ["L", "R", "L"]
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (- sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None, None, None, mode
    d2 = _mod2pi(2 * pi - acos(tmp))
    d1 = _mod2pi(-alpha - atan2(cos_a - cos_b, d + sin_a - sin_b) + d2 / 2.0)
    d3 = _mod2pi(_mod2pi(beta) - alpha - d1 + _mod2pi(d2))
    return d1, d2, d3, mode


_PATH_TYPE_MAP = {"LSL": _LSL, "RSR": _RSR, "LSR": _LSR, "RSL": _RSL,
                  "RLR": _RLR, "LRL": _LRL, }


def _dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                      step_size, planning_funcs):
    dx = end_x
    dy = end_y
    d = hypot(dx, dy) * curvature

    theta = _mod2pi(atan2(dy, dx))
    alpha = _mod2pi(-theta)
    beta = _mod2pi(end_yaw - theta)

    best_cost = float("inf")
    b_d1, b_d2, b_d3, b_mode = None, None, None, None

    for planner in planning_funcs:
        d1, d2, d3, mode = planner(alpha, beta, d)
        if d1 is None:
            continue

        cost = (abs(d1) + abs(d2) + abs(d3))
        if best_cost > cost:  # Select minimum length one.
            b_d1, b_d2, b_d3, b_mode, best_cost = d1, d2, d3, mode, cost

    lengths = [b_d1, b_d2, b_d3]
    x_list, y_list, yaw_list = _generate_local_course(lengths, b_mode,
                                                      curvature, step_size)

    lengths = [length / curvature for length in lengths]

    return x_list, y_list, yaw_list, b_mode, lengths, cost


def _interpolate(length, mode, max_curvature, origin_x, origin_y,
                 origin_yaw, path_x, path_y, path_yaw):
    if mode == "S":
        path_x.append(origin_x + length / max_curvature * cos(origin_yaw))
        path_y.append(origin_y + length / max_curvature * sin(origin_yaw))
        path_yaw.append(origin_yaw)
    else:  # curve
        ldx = sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - cos(length)) / -max_curvature
        gdx = cos(-origin_yaw) * ldx + sin(-origin_yaw) * ldy
        gdy = -sin(-origin_yaw) * ldx + cos(-origin_yaw) * ldy
        path_x.append(origin_x + gdx)
        path_y.append(origin_y + gdy)

        if mode == "L":  # left turn
            path_yaw.append(origin_yaw + length)
        elif mode == "R":  # right turn
            path_yaw.append(origin_yaw - length)

    return path_x, path_y, path_yaw


def _generate_local_course(lengths, modes, max_curvature, step_size):
    p_x, p_y, p_yaw = [0.0], [0.0], [0.0]

    for (mode, length) in zip(modes, lengths):
        if length == 0.0:
            continue

        # set origin state
        origin_x, origin_y, origin_yaw = p_x[-1], p_y[-1], p_yaw[-1]

        current_length = step_size
        while abs(current_length + step_size) <= abs(length):
            p_x, p_y, p_yaw = _interpolate(current_length, mode, max_curvature,
                                           origin_x, origin_y, origin_yaw,
                                           p_x, p_y, p_yaw)
            current_length += step_size

        p_x, p_y, p_yaw = _interpolate(length, mode, max_curvature, origin_x,
                                       origin_y, origin_yaw, p_x, p_y, p_yaw)

    return p_x, p_y, p_yaw

def backward_dijkstra(graph, goal):
    costs = np.zeros(graph.shape) + np.inf
    costs[goal[0], goal[1]] = 0
    open_queue = PriorityQueue()
    visited = np.zeros(map.shape, dtype=bool)
    open_queue.put((costs[goal[0], goal[1]], goal))
    actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
    while not open_queue.empty():
        current = open_queue.get()[1]
        if visited[current[0], current[1]]:
            continue
        visited[current[0], current[1]] = True
        neighbours = current + actions
        for n_node in neighbours:
            if not np.any((0 <= n_node[0] < graph.shape[0]) & (0 <= n_node[1] < graph.shape[1])) or visited[n_node[0], n_node[1]] or graph[n_node[0], n_node[1]] == 0:
                continue
            n_node_cost = costs[current[0], current[1]] + 1
            if n_node_cost < costs[n_node[0], n_node[1]]:
                costs[n_node[0], n_node[1]] = n_node_cost
                open_queue.put((n_node_cost, list(n_node)))
    return costs


def Image_to_use(url, is_bw=False):
    output = 'Map.png'
    gdown.download(url, output, quiet=False)
    image_colour = cv2.imread('Map.png')

    # convert the input image to grayscale
    image_bw = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)

    if not is_bw:
        # Convert to binary image
        ret, image_bw = cv2.threshold(image_bw, 170, 255, cv2.THRESH_BINARY)

    plt.imshow(image_bw, 'gray', vmin=0, vmax=255)
    plt.show()
    print(image_bw.shape)
    return image_colour, image_bw


def heuristic(node, goal):
    # return np.linalg.norm(np.array(goal) - np.array(node))  # euclidean distance
    # return abs(node[0] - goal[0]) + abs(node[1] - goal[1])  # manhattan distance
    # return BACKWARD_DIJKSTRA_COSTS[node[0], node[1]]        # backward dijkstra distance
    path_x, path_y, path_yaw, mode, lengths, cost = plan_dubins_path(node[0], node[1], 20, goal[0], goal[1], 30, 1)
    return cost                                               # dubins distance



def get_path(bp, goal, start):
    path = [goal]
    node = goal
    while list(node) != list(start):
        path.append(bp[tuple(node)])
        node = bp[tuple(node)]
    return path


def sanity_check(start, goal, map):
    if map[start[0], start[1]] == 0 or map[goal[0], goal[1]] == 0:
        return False
    else:
        return True

def Annotate(map, start, goal):
    # Draw a circle at pixel start with radius 5 and color red
    start = (start[1], start[0])
    goal = (goal[1], goal[0])
    cv2.circle(map, start, 5, (0, 0, 255), -1)
    # Write a caption
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(map, 'START', (start[0]-100, start[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Draw a circle at pixel goal with radius 5 and color red
    cv2.circle(map, goal, 5, (0, 0, 255), -1)
    # Write a caption
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(map, 'GOAL', goal, font, 1, (0, 0, 255), 2, cv2.LINE_AA)

def plot_path(image, path, map, visit):
    plot_map = np.copy(image)
    for i in range(len(map)):
        for j in range(len(map[i])):
            if visit[i, j]:
                plot_map[i, j] = [0, 0, 255]
    for point in path:
        plot_map[point[0], point[1]] = [255, 0, 0]
    # plt.imshow(plot_map)
    # plt.show()


def A_star(start, goal, map, video_map):
    if not sanity_check(start, goal, map):
        print('Goal or start node inside obstacle!')
        return []
    costs = np.zeros(map.shape) + np.inf
    costs[start[0], start[1]] = 0

    framesize = (map.shape[0], map.shape[1])
    out = cv2.VideoWriter('A_star_Euclidean.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, framesize)
    Annotate(video_map, start, goal)

    visited_nodes = np.zeros(map.shape, dtype=bool)

    open_nodes = PriorityQueue()
    open_nodes.put((costs[start[0], start[1]] + heuristic(start, goal), start))

    actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])

    back_pointers = dict()
    count = 0

    while open_nodes:
        count += 1
        curr_node = np.array(open_nodes.get()[1])
        if curr_node[0] == goal[0] and curr_node[1] == goal[1]:
            print('path found')
            print(np.count_nonzero(visited_nodes))
            print(costs[goal[0], goal[1]])
            final_path = get_path(back_pointers, curr_node, start)
            for point in final_path:
                video_map[point[0], point[1]] = [255, 0, 255]
            out.write(video_map)
            out.release()
            return final_path, visited_nodes

        video_map[curr_node[0], curr_node[1]] = [255, 0, 0]
        neighbours = curr_node + actions
        visited_nodes[curr_node[0], curr_node[1]] = True
        for n_node in neighbours:
            if not np.any((0 <= n_node[0] < map.shape[0]) & (0 <= n_node[1] < map.shape[1])) or visited_nodes[
                n_node[0], n_node[1]] or map[n_node[0], n_node[1]] == 0:
                continue
            n_node_cost = costs[curr_node[0], curr_node[1]] + 1
            if n_node_cost < costs[n_node[0], n_node[1]]:
                costs[n_node[0], n_node[1]] = n_node_cost
                back_pointers[tuple(n_node)] = curr_node
                open_nodes.put((n_node_cost + heuristic(n_node, goal), list(n_node)))
                video_map[n_node[0], n_node[1]] = [0, 0, 255]
        if count % 10 == 0:
            out.write(video_map)
    print('Path not found!')
    return []



Sunderbans = 'https://drive.google.com/uc?id=1XdwwfZ82tnWLOfkhrTXjS2lLI02c_ZGu'
start = [1000, 1000]
goal = [20, 630]
Archipelago = 'https://drive.google.com/uc?id=1QqfhtpBTHRpZiwhX6YUnoxUOksyemBoq'
# start = [149, 350]
# goal = [750, 139]
image, map = Image_to_use(Sunderbans, True)
# BACKWARD_DIJKSTRA_COSTS = backward_dijkstra(np.copy(map), goal)
path, visit = A_star(start, goal, map, np.copy(image))
print(path)
plot_path(image, path, map, visit)


