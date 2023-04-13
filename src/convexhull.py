
# yoinked from https://startupnextdoor.com/computing-convex-hull-in-python/
import math


class ConvexHull(object):

    def __init__(self, points=(), check_colinear=False):
        self._points = []
        self._hull_points = []
        self._check_colinear = check_colinear
        self.add_all(points)

    def add(self, point):
        self._points.append(point)

    def add_all(self, points):
        self._points.extend(points)

    def _get_orientation(self, origin, p1, p2):
        '''
        Returns the orientation of the Point p1 with regards to Point p2 using origin.
        Negative if p1 is clockwise of p2.
        :param p1:
        :param p2:
        :return: integer
        '''
        difference = (
            ((p2[0] - origin[0]) * (p1[1] - origin[1]))
            - ((p1[0] - origin[0]) * (p2[1] - origin[1]))
        )

        return difference

    def compute_hull(self):
        '''
        Computes the points that make up the convex hull.
        :return:
        '''
        points = self._points

        # get leftmost point
        start = points[0]
        min_x = start[0]
        for p in points[1:]:
            if p[0] < min_x:
                min_x = p[0]
                start = p

        point = start
        self._hull_points.append(start)

        far_point = None
        while far_point is not start:

            # get the first point (initial max) to use to compare with others
            p1 = None
            for p in points:
                if p is point:
                    continue
                else:
                    p1 = p
                    break

            far_point = p1

            for p2 in points:
                # ensure we aren't comparing to self or pivot point
                if p2 is point or p2 is p1:
                    continue
                else:
                    direction = self._get_orientation(point, far_point, p2)
                    if direction > 0:
                        far_point = p2

            self._hull_points.append(far_point)
            point = far_point

        if self._check_colinear:
            last = self._hull_points.pop()
            cx = sum(p[0] for p in self._hull_points) / len(self._hull_points)
            cy = sum(p[1] for p in self._hull_points) / len(self._hull_points)
            self._hull_points.sort(key=lambda p: math.atan2(p[0] - cx, p[1] - cy))
            self._hull_points = self._remove_colinears(self._hull_points)
            self._hull_points.append(last)



    def _remove_colinears(self, pts, start_i=0):
        if len(pts) <= 3:
            return pts
        else:
            for i in range(start_i, start_i + len(pts) + 2):
                i1 = i % len(pts)
                i2 = (i + 1) % len(pts)
                i3 = (i + 2) % len(pts)
                if self._is_colinear(pts[i1], pts[i2], pts[i3]):
                    pts.pop(i2)
                    return self._remove_colinears(pts, start_i=i1)
            return pts

    def _is_colinear(self, p1, p2, p3, thresh=0.0001):
        if abs(p1[0] - p2[0]) < thresh or abs(p2[0] - p3[0]) < thresh:
            return (p1[0] <= p2[0] <= p3[0] or p1[0] >= p2[0] >= p3[0]) and abs(p1[0] - p3[0]) < thresh
        else:
            s1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            s2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
            return abs(s2 - s1) < thresh

    def get_hull_points(self):
        if self._points and not self._hull_points:
            self.compute_hull()
        return self._hull_points
