from gym_duckietown.tasks.task_solution import TaskSolution

import numpy as np
import cv2


#==========BIRD EYE VIEW==================================================


def bird_eye_view(img):
    h, w = img.shape[:2]

    dx, dy = 297, 100
    src = np.float32([(     0, h-1),
                      (    dx,  dy),
                      (w-1-dx,  dy),
                      (   w-1, h-1)])

    h, w = 500, 300
    dx = 130
    dst = np.float32([(  0+dx, h-1),
                      (  0+dx,   0),
                      (w-1-dx,   0),
                      (w-1-dx, h-1)])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    
    return warped


#==========FIND COLOR==========================================================


def find_hsv_color(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


#==========CONNECTED COMPONENT=================================================


def check_cmp(mask, rect, size):    
    rect_size = (rect[2] - rect[0]) * (rect[3] - rect[1])
    if size / rect_size < 0.55:
        return False
    
    if size < 300:
        return False
    
    return True


def find_biggest_cmp(mask):
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    
    if numLabels == 1:
        return np.zeros_like(mask, dtype=np.uint8), None
    
    sizes = stats[1:, 4]
    i = sizes.argmax() + 1
    
    cmp = ((labels == i) * 255).astype(np.uint8)
    rect = stats[i, 0], stats[i, 1], stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]
    
    if not check_cmp(cmp, rect, stats[i, -1]):
        return np.zeros_like(mask, dtype=np.uint8), None
    
    return cmp, rect


#==========RANSAC==============================================================


def line_by_points(xs):
    x, y = xs[:, 0], xs[:, 1]
    
    A = np.vstack([x, np.ones_like(x)]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    
    line = np.array([m, -1, c])
    k = (line[0] ** 2 + line[1] ** 2) ** 0.5
    line = line / k
    
    return line


def get_liers(line, xs, thresh=20, inliers=True):
    n = xs.shape[0]
    
    ones = np.ones(n)[:, None]
    xs = np.concatenate([xs, ones], axis=1)
    
    dist = np.abs(xs @ line[:, None])
    dist = dist[:, 0]
    
    if inliers:
        ids = np.where(dist <= thresh)[0]
    else:
        ids = np.where(dist > thresh)[0]
    
    return ids


def ransac_try(xs, itr=10):
    n = xs.shape[0]
    ids = np.random.choice(n, 2, replace=False)
    
    for _ in range(itr):
        line = line_by_points(xs[ids])
        ids = get_liers(line, xs)
        
    return line, len(ids)


def ransac(xs, itr=10, inliers_tresh=300):
    if xs.shape[0] < inliers_tresh:
        return None

    best_line, best_inliers = None, 0
    for _ in range(itr):
        line, inliers = ransac_try(xs)
        
        if best_inliers < inliers:
            best_line, best_inliers = line, inliers
            
    if inliers < inliers_tresh:
        best_line = None
            
    return best_line


def find_lines_on_image(mask, n):
    mask = mask.T
    
    y, x = np.where(mask == 255)
    xs = np.vstack([x, y]).T
    
    lines = []
    for _ in range(n):
        line = ransac(xs)

        if line is not None:
            ids = get_liers(line, xs, inliers=False)
            xs = xs[ids]

            line = np.array([line[1], line[0], line[2]])
            
        lines += [line]
    
    return lines


#==========FIND DUCK===========================================================


def find_duck_color(img):
    lower = np.array([ 0, 200, 128])
    upper = np.array([50, 255, 255])
    return find_hsv_color(img, lower, upper)


def find_duck(img):
    mask = find_duck_color(img)
    cmp, rect = find_biggest_cmp(mask)
    return cmp, rect


#==========FIND BOARDER LINE===================================================


def find_boarder_line_color(img):
    lower = np.array([15,  0, 140])
    upper = np.array([50, 20, 180])
    return find_hsv_color(img, lower, upper)


def filter_boarder_line(mask):
    n = 7
    kernel = np.ones((n, n), dtype=np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)

    n = 10
    kernel = np.ones((n, n), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)
    
    return mask


def find_boarder_line(img):
    mask = find_boarder_line_color(img)
    mask = filter_boarder_line(mask)
    lines = find_lines_on_image(mask, n=2)
    return mask, lines


#==========FIND DOTTED LINE====================================================


def find_dotted_line_color(img):
    lower = np.array([ 0,  30,  90])
    upper = np.array([50, 230, 180])
    return find_hsv_color(img, lower, upper)


def find_dotted_line(img):
    mask = find_dotted_line_color(img)
    line = find_lines_on_image(mask, n=1)[0]
    return mask, line


#==========LOCALIZATION========================================================


def get_line_angle(line):
    if line is None:
        return None

    a, b, _ = line
    
    alpha = np.arctan2(b, a)
    
    if alpha < np.pi / 2:
        alpha += np.pi
    if alpha > np.pi / 2:
        alpha -= np.pi
        
    return alpha


def check_angles(a, b, delta=np.pi / 180 * 2):
    dif = abs(a - b)

    while dif > np.pi / 2:
        dif = abs(dif - np.pi)

    return dif < delta


def filter_lines(boarder_lines, dotted_line):
    l = boarder_lines[0]
    r = boarder_lines[1]
    m = dotted_line

    la = get_line_angle(l)
    ra = get_line_angle(r)
    ma = get_line_angle(m)

    if (la is not None) and (ra is not None):
        if not check_angles(la, ra):
            la, ra = None, None

    if ma is not None:
        if la is not None:
            if not check_angles(la, ma):
                la = None

        if ra is not None:
            if not check_angles(ra, ma):
                ra = None

    l = l if la is not None else None
    r = r if ra is not None else None
    m = m if ma is not None else None

    return [l, r], m


def get_average_angle(boarder_lines, dotted_line):
    l = get_line_angle(boarder_lines[0])
    r = get_line_angle(boarder_lines[1])
    m = get_line_angle(dotted_line)

    angles = [a for a in [l, r, m] if a is not None]
    if len(angles) > 0:
        return sum(angles) / len(angles)
    else:
        return None


def get_line_pos(l, center):
    if l is None:
        return None
    
    if l[0] < 0:
        l = -l
        
    return center[0] * l[0] + center[1] * l[1] + l[2]


def get_average_pos(boarder_lines, dotted_line, center):
    l = get_line_pos(boarder_lines[0], center)
    r = get_line_pos(boarder_lines[1], center)
    m = get_line_pos(dotted_line, center)
    
    if (l is not None) and (r is not None):
        if m is not None:
            return (l + m + r) / 3
        else:
            return (l + r) / 2
    else:
        return m


#==========SOLVER==============================================================


class DontCrushDuckieTaskSolution(TaskSolution):
    def __init__(self, generated_task):
        super().__init__(generated_task)

        self.env = self.generated_task['env']

    def proc_img(self, img):
        img = img[:, :, ::-1]

        biw = bird_eye_view(img)
    
        duck_cmp, duck_bb = find_duck(biw)
        
        boarder_line_mask, boarder_lines = find_boarder_line(biw)
        dotted_line_mask, dotted_line = find_dotted_line(biw)
        
        h, w = biw.shape[:2]
        center = np.array([w / 2, h])
        
        boarder_lines, dotted_line = filter_lines(boarder_lines, dotted_line)
        angle = get_average_angle(boarder_lines, dotted_line)
        pos = get_average_pos(boarder_lines, dotted_line, center)

        return duck_bb, angle, pos

    def go_to_duck(self):
        condition = True
        while condition:
            img, _, _, _ = self.env.step([1, 0])
            
            duck_bb, _, _ = self.proc_img(img)

            if duck_bb is None:
                condition = True
            else:
                condition = (duck_bb[3] < 500 - 10)

            self.env.render()

    def turn_left(self):
        for _ in range(33):
            _ = self.env.step([0, 1])
            self.env.render()

    def turn_right(self):
        for _ in range(33):
            _ = self.env.step([0, -1])
            self.env.render()

    def forward_n(self, n):
        for _ in range(n):
            _ = self.env.step([1, 0])
            self.env.render()

    def to_left_lane(self):
        self.turn_left()
        self.forward_n(n=7)
        self.turn_right()

    def to_right_lane(self):
        self.turn_right()
        self.forward_n(n=7)
        self.turn_left()

    def skip_duck(self):
        self.forward_n(n=15)

    def go_forward(self, n):
        img, _, _, _ = self.env.step([0, 0])

        for _ in range(n):
            _, angle, pos = self.proc_img(img)

            if pos is not None:
                pos = (pos - 32) / 32
                if pos > 1:
                    pos = 1
                if pos < -1:
                    pos = -1

                target_angle = pos / (np.pi / 6)

                va = (target_angle - angle) / np.pi
            elif angle is not None:
                va = -angle / np.pi
            else:
                va = 0

            img, _, _, _ = self.env.step([0.5, va])
            self.env.render()

    def stop(self):
        _ = self.env.step([0, 0])
        self.env.render()

    def solve(self):
        self.go_to_duck()
        self.to_left_lane()
        self.skip_duck()
        self.to_right_lane()

        for _ in range(10):
            _ = self.env.step([0, 1])
            self.env.render()
        # self.forward_n(n=3)

        self.go_forward(n=20)
        self.stop()
