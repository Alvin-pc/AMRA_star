import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import random
import heapq

# download the image files for mapping problem
import gdown


# url = 'https://drive.google.com/uc?id=1p4nuEn-T1v1ku3TO-Y3yY87Ro8SDWA7t'
# output = 'Sunderbans_map_1_color.png'
# gdown.download(url, output, quiet=False)
#
# image = cv2.imread('Sunderbans_map_1_color.png')

# # convert the input image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Convert to binary image
# ret, image_binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
#
# plt.imshow(image_binary, 'gray', vmin=0, vmax=255)
# # plt.show()
# print(image_binary.shape)

# output_binary = r'D:\Motion Planning Code\Sunderbans_map_bw.png'
# cv2.imwrite(output_binary, image_binary)

# url = 'https://drive.google.com/uc?id=1QptRsoEcFxJgmIXxDIR0YUs-lR7Co1mO'
# output = 'Sunderbans_map_1_color.png'
# gdown.download(url, output, quiet=False)
#
# image = cv2.imread('Sunderbans_map_1_color.png')
#
# # convert the input image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Convert to binary image
# ret, image_binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
#
# plt.imshow(image_binary, 'gray', vmin=0, vmax=255)
# plt.show()
# print(image_binary.shape)

def Image_to_use(url, is_bw=False):
    output = 'Map.png'
    gdown.download(url, output, quiet=False)
    image_colour = cv2.imread('Map.png')

    # convert the input image to grayscale
    image_bw = cv2.cvtColor(image_colour, cv2.COLOR_BGR2GRAY)

    if not is_bw:
        # Convert to binary image
        ret, image_bw = cv2.threshold(image_bw, 170, 255, cv2.THRESH_BINARY)
        # image_bw = cv2.adaptiveThreshold(image_bw, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)

    plt.imshow(image_bw, 'gray', vmin=0, vmax=255)
    plt.show()
    print(image_bw.shape)
    return image_colour, image_bw


class MyPriorityQueue():
    def __init__(self):
        self.p_queue = dict()

    def put(self, element):
        self.p_queue[tuple(element[1])] = element

    def copy(self):
        self.p_queue.copy()

    def get(self):
        # min_key = min(self.p_queue, key=self.p_queue.get)
        # min_value = self.p_queue[min_key]
        min_value = heapq.nsmallest(1, self.p_queue.values())
        # print(min_value)
        self.p_queue.pop(tuple(min_value[0][1]))

        return min_value[0]

    def update(self, x, key):
        self.p_queue[tuple(x)] = (key, x)

    def remove(self, element):
        self.p_queue.pop(tuple(element))

    def min(self):
        # min_key = min(self.p_queue, key=self.p_queue.get)
        # min_value = self.p_queue[min_key]
        min_value = heapq.nsmallest(1, self.p_queue.values())

        return min_value[0][0]

    def empty(self):
        if len(list(self.p_queue.keys())) == 0:
            return True
        else:
            return False

    def get_elements(self):
        return list(self.p_queue.values())

    def contains(self, key):
        return tuple(key) in list(self.p_queue.keys())

    def length(self):
        return len(self.p_queue)


class AMRA_Star:
    def __init__(self, map, color_map, resolutions, heuristics, heuristic_to_res, start, goal, size_of_UAV, w1=500, w2=500):
        # sanity check
        self.sanity = True
        if map[start[0], start[1]] == 0 or map[goal[0], goal[1]] == 0:
            print('Start or goal node inside an obstacle !')
            self.sanity = False
            # return

        while map[start[0], start[1]] == 0:
            start = random.sample(range(map.shape[0]), 2)
        while map[goal[0], goal[1]] == 0:
            goal = random.sample(range(map.shape[0]), 2)

        self.map = map
        self.color_map = color_map
        self.video_map = np.copy(color_map)
        self.cmap = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'magenta': (255, 0, 255),
            'cyan': (255, 255, 0),
            'pink': (147, 20, 255)
        }

        self.list_of_resolutions = resolutions  # list of resolution tuples
        self.list_of_heuristics = heuristics  # list of heuristic function handles - takes node, goal as input -> output is heuristic value
        self.start = start
        self.goal = goal

        self.w1 = w1
        self.w2 = w2

        self.buffer = Extended_UAV(size_of_UAV[0], size_of_UAV[1])

        self.OPEN = dict()  # dict of priority queues indexed by heuristic
        self.CLOSED = dict()  # dict of closed lists indexed by resolution
        self.INCONS = []  # list of inconsistent states

        self.obtained_paths = []  # list to store obtained paths

        self.current_queue_index = 0

        self.back_pointers = dict()  # dict of backpointers indexed by tuple form of node -> value is the backpointer node
        self.heuristic_to_res = heuristic_to_res  # dict mapping heuristic indices to resolution indices
        self.g = np.zeros(map.shape) + np.inf  # matrix of costs for each state

        self.num_expansions = 0

        self.framesize = (self.color_map.shape[0], self.color_map.shape[1])
        # self.out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), 60, self.framesize)

        self.init_search()

    def init_search(self):
        for i in range(len(self.list_of_resolutions)):
            self.CLOSED[i] = []

        for i in range(len(self.list_of_heuristics)):
            self.OPEN[i] = MyPriorityQueue()

    def obstacles_in_range(self, node, res_idx):
        resolution = self.list_of_resolutions[res_idx]
        x_diff = int((self.list_of_resolutions[res_idx][0] - 1) / 2)
        y_diff = int((self.list_of_resolutions[res_idx][1] - 1) / 2)

        # keep neighbour checks also within map
        x_lower = node[0] - x_diff - self.buffer[0]
        x_upper = node[0] + x_diff + 1 + self.buffer[0]
        if x_lower < 0:
            x_lower = 0
        if x_upper > self.map.shape[0]:
            x_upper = self.map.shape[0]

        y_lower = node[1] - y_diff - self.buffer[1]
        y_upper = node[1] + y_diff + 1 + self.buffer[1]
        if y_lower < 0:
            y_lower = 0
        if y_upper > self.map.shape[1]:
            y_upper = self.map.shape[1]

        snap = self.map[x_lower:x_upper, y_lower:y_upper]
        if 0 in snap:
            return True
        else:
            return False

    def Succs(self, x, res_idx):
        list_of_res = [res_idx]
        if res_idx == 0:
            list_of_res = self.Resolutions(x)
        succs = []
        for resolution_idx in list_of_res:
            jump_x = self.list_of_resolutions[resolution_idx][0]
            jump_y = self.list_of_resolutions[resolution_idx][1]
            # succs = []

            actions = np.array([[-jump_x, 0], [jump_x, 0], [0, jump_y], [0, -jump_y]])
            neighbours = list(np.array(x) + actions)
            # check the anchor search part from the paper
            for n_node in neighbours:
                if not np.any((0 + self.buffer[0] <= n_node[0] < self.map.shape[0] - self.buffer[0]) & (
                        0 + self.buffer[1] <= n_node[1] < self.map.shape[1] - self.buffer[1])) or self.obstacles_in_range(n_node, resolution_idx):
                    # or self.map[n_node[0], n_node[1]] == 0:
                    # or self.obstacles_in_range(n_node, resolution_idx):
                    continue
                succs.append(tuple(n_node))
        return succs

    def ChooseQueue(self):  # round-robin over inadmissible queues -> 1 to N
        if len(self.list_of_heuristics) == 1:  # edge case of one heuristic and one res only
            return 0
        if self.current_queue_index == len(self.list_of_heuristics) - 1:
            return 1
        else:
            self.current_queue_index += 1
            return self.current_queue_index

    def Res(self, i):
        return self.heuristic_to_res[i]

    def Resolutions(self, x):
        res = []
        for i in range(len(self.list_of_resolutions)):
            jump_x = self.list_of_resolutions[i][0]
            jump_y = self.list_of_resolutions[i][1]

            if x[0] % jump_x == 0 and x[1] % jump_y == 0:
                res.append(i)
        return res  # return resolution indices

    def cost(self, x, x_prime):
        return abs(x[0] - x_prime[0]) + abs(x[1] - x_prime[1])  # Manhattan

    def Key(self, x, i):
        return self.g[x[0], x[1]] + self.w1 * self.list_of_heuristics[i](x, self.goal)

    def get_path(self):
        path = [self.goal]
        node = self.goal
        while list(node) != list(self.start):
            path.append(self.back_pointers[tuple(node)])
            node = self.back_pointers[tuple(node)]
        return path

    def plot_path(self, path):
        plot_map = np.copy(self.color_map)
        # for CLOSED_i in self.CLOSED.values():
        #     if len(CLOSED_i) > 0:
        #         for point in CLOSED_i:
        #             plot_map[point[0], point[1]] = [0, 0, 255]
        for point in path:
            plot_map[range(point[0]-1, point[0]+2), range(point[1]-1, point[1]+2)] = [255, 0, 0]
        Annotate(plot_map, self.start, self.goal, self.w1, self.w2)
        plt.figure()
        plt.imshow(plot_map)
        plt.show()
        output_pics = f'Sunderbans_expanded_w={self.w1}.png'
        cv2.imwrite(output_pics, plot_map)

    def get_paths(self):
        return self.obtained_paths

    def backward_dijkstra_calculator(self, goal):
        costs = np.zeros(self.map.shape) + np.inf
        costs[goal[0], goal[1]] = 0
        open_queue = MyPriorityQueue()
        visited = np.zeros(map.shape, dtype=bool)
        open_queue.update(goal, 0)
        actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
        while not open_queue.empty():
            current = open_queue.get()[1]
            if visited[current[0], current[1]]:
                continue
            visited[current[0], current[1]] = True
            neighbours = current + actions
            for n_node in neighbours:
                if not np.any((0 <= n_node[0] < self.map.shape[0]) & (0 <= n_node[1] < self.map.shape[1])) or visited[
                    n_node[0], n_node[1]] or self.map[n_node[0], n_node[1]] == 0:
                    continue
                n_node_cost = costs[current[0], current[1]] + 1
                if n_node_cost < costs[n_node[0], n_node[1]]:
                    costs[n_node[0], n_node[1]] = n_node_cost
                    open_queue.update(list(n_node), n_node_cost)
        return costs

    def Expand(self, x, i):
        col = 255 - int((max(self.w1,self.w2) - 1) * 0.4)
        self.num_expansions += 1
        self.video_map[x[0], x[1]] = [col, 255, 0]
        if self.num_expansions % 10000 == 0:
            print(self.num_expansions)
            print(self.OPEN[0].length())
        r = self.Res(i)
        if i != 0:
            for j in range(1, len(self.list_of_heuristics)):
                if j != i and self.Res(j) == r:
                    # self.OPEN[j].remove(x)
                    if self.OPEN[j].contains(
                            x):  # added this for the case where 2nd heuristic is greater than w2 bounds in the resolution. This is not handled in the paper
                        self.OPEN[j].remove(x)

        for x_prime in self.Succs(x, r):
            if self.g[x_prime[0], x_prime[1]] > self.g[x[0], x[1]] + self.cost(x, x_prime):
                self.g[x_prime[0], x_prime[1]] = self.g[x[0], x[1]] + self.cost(x, x_prime)
                self.back_pointers[tuple(x_prime)] = list(x)
                self.video_map[x_prime[0], x_prime[1]] = [0, 255, col]
                if x_prime in self.CLOSED[0]:
                    self.INCONS.append(x_prime)
                else:
                    self.OPEN[0].update(x_prime, self.Key(x_prime, 0))
                    for j in range(1, len(self.list_of_heuristics)):
                        l = self.Res(j)
                        if l not in self.Resolutions(x_prime):
                            continue
                        if x_prime not in self.CLOSED[l]:
                            if self.Key(x_prime, j) <= self.w2 * self.Key(x_prime, 0):
                                self.OPEN[j].update(x_prime, self.Key(x_prime, j))

    def ImprovePath(self):
        open_empty = np.all([self.OPEN[i].empty() for i in range(len(list(self.OPEN.keys())))])
        i = self.current_queue_index
        while not open_empty:
            while self.OPEN[i].empty():
                i = self.ChooseQueue()
            if self.OPEN[i].min() > self.w2 * self.OPEN[0].min():
                i = 0

            x = self.OPEN[i].get()[1]
            self.Expand(x, i)
            # self.out.write(self.video_map)
            r = self.Res(i)
            self.CLOSED[r].append(x)
            # print(x)
            self.ChooseQueue()
            if tuple(x) == tuple(self.goal):
                return True

    def Main(self):
        if not self.sanity:
            print('Start/Goal chosen randomly')
            # return

        self.g[self.start[0], self.start[1]] = 0
        self.INCONS.append(tuple(self.start))
        self.current_queue_index = 1

        try:
            while self.w1 >= 500 and self.w2 >= 500:
                # self.video_map = np.copy(self.color_map)
                self.num_expansions = 0
                for x in self.INCONS:
                    self.OPEN[0].update(x, self.Key(x, 0))

                self.INCONS = []

                for x in self.OPEN[0].get_elements():
                    Res_of_x = self.Resolutions(x[1])
                    for j in range(1, len(self.list_of_heuristics)):
                        if self.Res(j) in Res_of_x:
                            self.OPEN[j].update(x[1], self.Key(x[1], j))

                for i in range(len(self.CLOSED)):
                    self.CLOSED[i] = []

                if self.ImprovePath():
                    path = self.get_path()
                    for point in path:
                        robotx = range((point[0] - self.buffer[0]),(point[0] + self.buffer[0] + 1))
                        roboty = range((point[1] - self.buffer[1]), (point[1] + self.buffer[1] + 1))
                        self.video_map[robotx, roboty] = [255, 0, 255]
                    # for i in range(20):
                        # self.out.write(self.video_map)
                    print('Path Found at w1 = ' + str(self.w1) + ' and w2 = ' + str(self.w2) + ':', str(path))
                    self.plot_path(path)
                    self.obtained_paths.append(path)
                else:
                    print('No Path Found at w1 = ' + str(self.w1) + ' and w2 = ' + str(self.w2))

                # self.out.release()
                self.w1 -= 0.75 * (self.w1 - 1)
                self.w2 -= 0.75 * (self.w2 - 1)

                self.w1 = round(self.w1, 1)
                self.w2 = round(self.w2, 1)
        except KeyboardInterrupt:
            print("Interuppt")
            # self.out.release()
        # self.out.release()


def Annotate(map, start, goal, w1_val, w2_val):
    # Draw a circle at pixel start with radius 5 and color red
    start = (start[1], start[0])
    goal = (goal[1], goal[0])
    cv2.circle(map, start, 5, (0, 0, 255), -1)
    # Write a caption
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(map, 'START', (start[0] - 100, start[1]), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # Draw a circle at pixel goal with radius 5 and color red
    cv2.circle(map, goal, 5, (0, 0, 255), -1)
    cv2.putText(map, 'GOAL', goal, font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(map, f'w1= {w1_val}, w2= {w2_val}', (200, 1000), font, 1, (0, 0, 255), 2, cv2.LINE_AA)



def euclidean(node, goal):
    return np.linalg.norm(np.array(goal) - np.array(node))  # euclidean distance


def manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def backward_dijkstra(node, goal):
    return Cost_map[goal[0], goal[1]]

def Extended_UAV(size_x, size_y):
    buffer = (int((size_x - 1)/2), int((size_y - 1)/2))
    return buffer

Sunderbans = 'https://drive.google.com/uc?id=1p4nuEn-T1v1ku3TO-Y3yY87Ro8SDWA7t'
# start = [1000, 1000]
# goal = [20, 630]
Across_the_Cape = 'https://drive.google.com/uc?id=1QptRsoEcFxJgmIXxDIR0YUs-lR7Co1mO'
Archipelago = 'https://drive.google.com/uc?id=1QqfhtpBTHRpZiwhX6YUnoxUOksyemBoq'
# start = [149, 350]
# goal = [750, 139]


image, map = Image_to_use(Archipelago)
resolutions = [(1, 1), (1, 1), (3, 3), (9, 9)]
heuristics = [euclidean, euclidean, manhattan, manhattan]
heuristic_to_res = {0: 0, 1: 1, 2: 2, 3: 3}
start = [149, 350]
goal = [750, 139]
size_of_UAV = (1, 1)
amra_star_solver = AMRA_Star(map, image, resolutions, heuristics, heuristic_to_res, start, goal, size_of_UAV, w1=501, w2=501)
# Cost_map = amra_star_solver.backward_dijkstra_calculator(goal)
amra_star_solver.Main()
print('done')
