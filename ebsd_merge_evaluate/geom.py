"""
Geometry operations on lines and circles.
"""

from math import isclose
import numpy as np


class Line:
    """
    An infinite line defined by  a*x + b*y = c
    """

    def __init__(
        self,
        a=None,
        b=None,
        c=None,
        point1=None,
        point2=None,
    ):

        if a is None or b is None or c is None:
            if not (a is None and b is None and c is None):
                raise ValueError(
                    "If any a, b, c is unspecified, " "all of them must be unspecified."
                )
            if point1 is None or point2 is None:
                raise ValueError(
                    "If any of a, b, c is unspecified, " "x1 and x2 must be specified."
                )
            point1 = np.asarray(point1)
            point2 = np.asarray(point2)
            if point1.size != 2 or point2.size != 2:
                raise ValueError("x1 and x2 must be size 2 array-like.")
            a = point2[1] - point1[1]
            b = point1[0] - point2[0]
            c = point1[0] * point2[1] - point2[0] * point1[1]
        else:
            if not (point1 is None and point2 is None):
                raise ValueError(
                    "If a, b, c are specified, " "x1 and x2 must not be specified."
                )

        self.a = a
        self.b = b
        self.c = c

        self.steep = abs(a) > abs(b)
        if self.steep:  # x = k*y + d
            self.k = -b / a
            self.d = c / a
        else:  # y = k*x + d
            self.k = -a / b
            self.d = c / b

    def isbelow(
        self,
        point,
        tol=0,
    ):
        """
        Determine if line is below point.

        :param point: 2-tuple containing the coordinates of the point.
        :param tol: absolute tolerance (negative if more stringent)
        :return: flag indicating if line is below point.
        """
        x, y = point
        if self.steep:  # x = k*y + d
            if self.k == 0:
                return False
            else:
                return (x - self.d) / self.k < y + tol
        else:
            return self.k * x + self.d < y + tol

    def isabove(
        self,
        point,
        tol=0,
    ):
        """
        Determine if line is below point.

        :param point: 2-tuple containing the coordinates of the point.
        :param tol: absolute tolerance (negative if more stringent)
        :return: flag indicating if line is above point.
        """
        return not self.isbelow(point, -tol)


class Circle:
    """
    A circle defined by  (x-xc)**2 + (y-yc)**2 = r**2
    """

    def __init__(
        self,
        xc,
        yc,
        r,
    ):
        self.xc = xc
        self.yc = yc
        self.r = r

    def contains(
        self,
        point,
        tol=0.001,
    ):
        """
        Determine whether circle contains point.

        In other words, if point is inside or on the outline

        :param point: 2-tuple containing the coordinates of the point.
        :param tol: absolute tolerance.
        :return: flag indicating if circle contains point.
        """
        x, y = point
        return (x - self.xc) ** 2 + (y - self.yc) ** 2 <= (self.r + tol) ** 2

    def get_polygon(
        self,
        angle1=0,
        angle2=2 * np.pi,
        dangle=np.radians(1),
    ):
        """
        Return circle or arc as a polygon.

        The points are ordered counter-clockwise. If angle2 < angle1, the arc
        is assumed between angle1 and angle2+2*pi.

        :param angle1: start angle (rad)
        :param point2: final angle (rad)
        :param dangle: angle increment (rad)
        :return xy: arc as a polygon (Nx2 array)
        """
        if angle2 < angle1:
            angle2 += 2 * np.pi
        if isclose(angle2 - angle1, 2 * np.pi):
            angle2 -= dangle
        angles = np.linspace(
            angle1,
            angle2,
            int((angle2 - angle1) / np.radians(dangle) + 1),
            endpoint=True,
        )
        x = self.xc + self.r * np.cos(angles)
        y = self.yc + self.r * np.sin(angles)

        xy = np.vstack((x, y)).T

        return xy


class Arc:
    """
    Arc of circle between the radii through point1 and point2.

    Always the shorter of the two possible arcs (<180 deg) is taken.
    The orientation of the arc in counterclockwise.
    """

    def __init__(
        self,
        xc,
        yc,
        r,
        point1,
        point2,
    ):
        self.xc = xc
        self.yc = yc
        self.r = r
        x1, y1 = point1
        x2, y2 = point2
        self.angle1 = np.arctan2(y1 - self.yc, x1 - self.xc)
        self.angle2 = np.arctan2(y2 - self.yc, x2 - self.xc)
        # make sure between angles < 180 deg
        dangle = self.angle2 - self.angle1
        if -np.pi < dangle < 0 or dangle > np.pi:
            self.angle1, self.angle2 = self.angle2, self.angle1

    def get_polygon(
        self,
        dangle=np.radians(0.1),
    ):
        """
        Return arc as a polygon.

        :param dangle: angle increment (rad)
        :return xy: arc as a polygon (Nx2 array)
        """
        angle1 = self.angle1
        angle2 = self.angle2
        if angle2 < angle1:
            angle2 += 2 * np.pi
        angles = np.linspace(
            angle1, angle2, (angle2 - angle1) / dangle + 1, endpoint=True
        )
        x = self.xc + self.r * np.cos(angles)
        y = self.yc + self.r * np.sin(angles)

        xy = np.vstack((x, y)).T

        return xy


def _intersect_line_line(
    line1,
    line2,
):
    """
    Intersect two straight lines.

    If the lines are parallel, no value is returned.
    :param line1 (Line): Line 1.
    :param line2 (Line): Line 2.
    :return (x, y): Intersection point or nothing.
    """
    denom = line1.a * line2.b - line2.a * line1.b
    if denom == 0:
        return []
    x = (line1.b * line2.c - line2.b * line1.c) / (-denom)
    y = (line1.a * line2.c - line2.a * line1.c) / denom

    return [(x, y)]


def _intersect_circle_line(
    circle,
    line,
):
    """
    Determine intersection points of a circle and a line.

    :param circle: Circle.
    :param line: Line.
    :return [(x, y), ...]: List of intersection points.
    """
    if line.steep:  # x = k*y + d
        a = line.k**2 + 1
        b = -2 * (circle.yc + line.k * (circle.xc - line.d))
        c = circle.yc**2 + (circle.xc - line.d) ** 2 - circle.r**2
        ys = np.roots([a, b, c])
        ys = ys[np.isreal(ys)]
        xs = ys * line.k + line.d
        return [(x, y) for x, y in zip(xs, ys)]
    else:  # y = k*x + d
        a = line.k**2 + 1
        b = -2 * (circle.xc + line.k * (circle.yc - line.d))
        c = circle.xc**2 + (circle.yc - line.d) ** 2 - circle.r**2
        xs = np.roots([a, b, c])
        xs = xs[np.isreal(xs)]
        ys = xs * line.k + line.d
        return [(x, y) for x, y in zip(xs, ys)]


def _intersect_circle_circle(
    circle1,
    circle2,
):
    """
    Determine intersection points of two circles.

    :param circle1: Circle 1
    :param circle2: Circle 2
    :return [(x, y), ...]: List of intersection points.
    """
    a = 2 * (circle2.xc - circle1.xc)
    b = 2 * (circle2.yc - circle1.yc)
    c = (circle2.xc**2 + circle2.yc**2 - circle2.r**2) - (
        circle1.xc**2 + circle1.yc**2 - circle1.r**2
    )
    line = Line(a, b, c)

    return _intersect_circle_line(circle1, line)


def intersect(
    object1,
    object2,
):
    """
    Intersect two geometric objects (circles or lines).

    :param object1 (Circle or Line): Object 1.
    :param object2 (Circle or Line): Object 2.
    :return (x, y): Intersection point(s).
    """
    if isinstance(object1, Line) and isinstance(object2, Line):
        return _intersect_line_line(object1, object2)
    elif isinstance(object1, Line) and isinstance(object2, Circle):
        return _intersect_circle_line(object2, object1)
    elif isinstance(object1, Circle) and isinstance(object2, Line):
        return _intersect_circle_line(object1, object2)
    elif isinstance(object1, Circle) and isinstance(object2, Circle):
        return _intersect_circle_circle(object1, object2)


if __name__ == "__main__":

    line1 = Line(1, 4, 6)
    line2 = Line(-5, 2, -8)

    line3 = Line(point1=(2, 1), point2=(6, 0))
    assert isclose(line3.k, line1.k)
    assert isclose(line3.d, line1.d)

    circle1 = Circle(-1, 2, 5)
    circle2 = Circle(-2, 0, 4)

    solutions = intersect(line1, line2)
    for solution in solutions:
        x, y = solution
        print("x={}, y={}".format(x, y))
        assert isclose(x, 2)
        assert isclose(y, 1)

    solutions = intersect(circle1, line1)
    for solution in solutions:
        x, y = solution
        print("x={}, y={}".format(x, y))
        assert isclose((x - circle1.xc) ** 2 + (y - circle1.yc) ** 2, circle1.r**2)
        assert isclose(line1.a * x + line1.b * y, line1.c)

    solutions = intersect(line2, circle1)
    for solution in solutions:
        x, y = solution
        print("x={}, y={}".format(x, y))
        assert isclose((x - circle1.xc) ** 2 + (y - circle1.yc) ** 2, circle1.r**2)
        assert isclose(line2.a * x + line2.b * y, line2.c)

    solutions = intersect(circle1, circle2)
    for solution in solutions:
        x, y = solution
        print("x={}, y={}".format(x, y))
        assert isclose((x - circle1.xc) ** 2 + (y - circle1.yc) ** 2, circle1.r**2)
        assert isclose((x - circle2.xc) ** 2 + (y - circle2.yc) ** 2, circle2.r**2)
