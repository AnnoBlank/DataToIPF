"""
Geometry operations with a wedge.

A wedge is defined by the triangle bounded by the lines y=0 and y=x, and by
a circle with its center on the negative x axis or at the origin.
"""

import numpy as np

from ebsd_merge_evaluate.geom import Line, Circle, intersect
from ebsd_merge_evaluate.stereographic import stereographic_projection, cartesian


class Wedge:
    """
    Wedge defined as described above. The circle is defined upon
    initialization by the stereographic projection of the plane spanned by two
    crystal axes, or by a radius. In the latter case, the center of the
    circle is assumed at the origin.
    """

    def __init__(
        self,
        dir1=(1, 0, 1),
        dir2=(1, 1, 1),
        theta_max=None,
    ):
        self.horizontal_line = Line(0, 1, 0)
        self.diagonal_line = Line(1, -1, 0)
        if theta_max is None:
            dirn = np.cross(dir1, dir2)
            if dirn[2] < 0:
                dirn = -dirn
            costheta = dirn[2] / np.linalg.norm(dirn)
            phi = np.arctan2(dirn[1], dirn[0])
            r = 2 / costheta
            rc = r * np.sqrt(1 - costheta**2)
            xc, yc = cartesian(rc, phi)
            self.circle = Circle(xc, yc, r)
        else:
            rmax = stereographic_projection(theta_max)
            self.circle = Circle(0, 0, rmax)

    def contains(
        self,
        point,
        tol=0.001,
    ):
        """
        Determine if wedge contains the point.

        :param point: Point.
        :return: Flag indicating whether wedge contains the point.
        """
        return (
            self.horizontal_line.isbelow(point, tol)
            and self.diagonal_line.isabove(point, tol)
            and self.circle.contains(point, tol)
        )

    def intersect(
        self,
        object,
        tol=0.001,
    ):
        """
        Intersect the wedge with a line or circle.

        :param object: Line or circle to be intersected.
        :param tol: absolute tolerance for containing points or
            removing duplicate points.
        :return [(x, y), ...]: List of intersection points.
        """
        # intersect the lines and the circle defining the wedge with the line
        points = list(intersect(self.horizontal_line, object))
        points += list(intersect(self.diagonal_line, object))
        points += list(intersect(self.circle, object))
        # collect points outside the wedge
        delete_points = set()
        for point in points:
            if not self.contains(point):
                delete_points.add(point)
        # collect duplicate points
        points = list(set(points))
        for point1 in points:
            for point2 in points:
                if point1 is point2:
                    break
                x1, y1 = point1
                x2, y2 = point2
                if abs(x1 - x2) < tol and abs(y1 - y2) < tol:
                    delete_points.add(point1)
        # remove points
        for delete_point in delete_points:
            points.remove(delete_point)

        # order according to increasing distance from origin
        def radius(point):
            return np.hypot(*point)

        points.sort(key=radius)
        return points

    def get_polygon(
        self,
        dangle=np.radians(1),
    ):
        """
        Get the wedge as a polygon.

        :param dphi: Approximate phi increment of the circle part (deg)
        :return xy: wedge as a polygon (Nx2 array)
        """
        # end points and angles of arc
        points = intersect(self.horizontal_line, self.circle)
        if self.diagonal_line.isabove(points[0]):
            point1 = points[0]
        else:
            point1 = points[1]
        angle1 = np.arctan2(point1[1] - self.circle.yc, point1[0] - self.circle.xc)
        points = intersect(self.diagonal_line, self.circle)
        if self.horizontal_line.isbelow(points[0]):
            point2 = points[0]
        else:
            point2 = points[1]
        angle2 = np.arctan2(point2[1] - self.circle.yc, point2[0] - self.circle.xc)

        # arc
        xy = self.circle.get_polygon(angle1, angle2, dangle)

        # prepend origin
        origin = intersect(self.horizontal_line, self.diagonal_line)
        xy = np.insert(xy, 0, origin[0], axis=0)

        return xy


if __name__ == "__main__":
    from math import isclose

    wedge = Wedge()
    print(wedge.circle.xc, wedge.circle.yc, wedge.circle.r)
    print(wedge.get_polygon())

    wedge = Wedge((1, 1, 2), (1, 0, 1))
    print(wedge.circle.xc, wedge.circle.yc, wedge.circle.r)
    print(wedge.get_polygon())

    exit()

    line1 = Line(1, 4, 6)
    line2 = Line(-5, 2, -8)
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
