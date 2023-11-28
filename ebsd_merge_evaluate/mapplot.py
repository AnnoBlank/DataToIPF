"""
Plot functions for channeling maps.
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ebsd_merge_evaluate.geom import Line, Circle, Arc
from ebsd_merge_evaluate.stereographic import stereographic_projection, cartesian, inverse_stereographic_projection
from ebsd_merge_evaluate.wedge import Wedge


def loc2align(loc):
    """
    Helper function to convert relative location to alignment.

    Location is the placement of the object relative to the anchor; Alignment
    is the position of the anchor relative to the object.

    :param loc: '<vloc> <hloc>', where <vloc> is one of 'bottom', 'center',
        'top', and <hloc> is one of 'left', 'center', 'right'. <hloc> may be
        omitted.
    :return (va, ha): vertical and horizontal alignment.
    """
    loc = loc.split()
    vloc = loc[0]
    if len(loc) > 1:
        hloc = loc[1]
    else:
        hloc = 'center'

    va = {'top': 'bottom', 'center': 'center', 'bottom': 'top'}
    va = va[vloc]
    ha = {'left': 'right', 'center': 'center', 'right': 'left'}
    ha = ha[hloc]

    return va, ha


def plot_text(text, position, ha='center', va='center', offset=0, rotation=0,
              fig=None, ax=None):
    """
    Plot text at given position with offset an rotation.

    :param text: Text to be plotted.
    :param position: 2-tuple specifying the (x,y) position.
    :param ha: horizontal alignment ('left', 'center', or 'right').
    :param va: vertical alignment ('top', 'center', or 'bottom').
    :param offset: offset of text anchor point from position both in text
        rotation direction and perpendicular to it in units of the font size.
    :param rotation: rotation anlge (deg) counterclockwise relative to
        horizontal.
    :param fig: Figure to be plotted to.
    :param ax: Axes to be plotted to.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    dirt = np.array((np.cos(np.radians(rotation)),
                     np.sin(np.radians(rotation))))
    dirn = np.array((- dirt[1], dirt[0]))

    # shift anchor point
    fontsize = mpl.rcParams['font.size']
    shift = offset * fontsize
    dh = {'left': shift, 'center': 0, 'right': -shift}
    dv = {'bottom': shift, 'center': 0, 'top': -shift}
    dh = dh[ha]
    dv = dv[va]
    shift = dh * dirt + dv * dirn
    dx, dy = shift
    transform = mpl.transforms.offset_copy(ax.transData, fig,
                                           dx, dy, units = 'points')

    # plot the text
    ax.text(*position, text, ha=ha, va=va, transform=transform,
            rotation=rotation, size=20)


def plot_direction(miller, marker='+', loc='center', offset=0.5,
                   fig=None, ax=None):
    """
    Mark a direction given by Miller indices.

    :param miller: 3-tuple containing the Miller indices.
    :param marker: Symbol to be plotted at direction.
        An empty string or None means no symbol to be plotted.
    :param loc: Position of annotation. Valid choices are those that can
        also be passed to legend() by the loc argument. An empty string or None
        means no annotation.
    :param offset: Offset of the annotation in units of the font size.
        Not applied if pos='center'.
    :param fig: Figure to be plotted to
    :param ax: Axes to be plotted to.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    dir = np.asarray(miller, dtype=float)
    dir /= np.linalg.norm(dir)
    theta = np.arccos(dir[2])
    phi = np.arctan2(dir[1], dir[0])
    r = stereographic_projection(theta)
    x, y = cartesian(r, phi)

    if marker:
        ax.plot((x,), (y,), 'k', marker=marker)

    if loc:
        text = '['
        for m in miller:
            if m >= 0:
                text += str(m)
            else:
                text += r'$\overline{' + str(-m) + r'}$'
        text += ']'

        va, ha = loc2align(loc)

        plot_text(text, (x, y), ha, va, offset)


def plot_plane(miller, frame, linestyle='-', loc='center',
               offset=0, fig=None, ax=None):
    """
    Show the plane by drawing a line.

    :param miller: 3-tuple containing the Miller indices.
    :param frame: Frame object to which the line representing the plane is to
        be clipped (Wedge).
    :param linestyle: Line style (if empty or None, no line is drawn)
    :param loc: Position of annotation. Valid choices are 'top',
        'center', and 'bottom'
    :param offset: Offset of the annotation in units of the font size.
        Not applied if pos='center' (if empty or None, no annotation is
        written).
    :param fig: Figure to be plotted to
    :param ax: Axes to be plotted to.
   """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    normal = np.asarray(miller, dtype=float)
    normal /= np.linalg.norm(normal)
    straight = (normal[2] == 0)

    # parameters describing the plane
    if straight:
        phi = - np.arctan2(normal[0], normal[1])
        plane = Line(point1=(0,0), point2=(np.cos(phi),np.sin(phi)))
    else:
        r = - 2 / normal[2]
        phic = np.arctan2(normal[1], normal[0])
        rc = - r * np.hypot(normal[0], normal[1])
        xc, yc = cartesian(rc, phic)
        plane = Circle(xc, yc, r)

    # intersections with frame
    points = frame.intersect(plane)
    if len(points) != 2:
        print(points)
        print('plot_plane: skipping ' + str(miller))
        return

    # text position and rotation
    if straight:
        xy = np.array(points)

        dirt = xy[1] - xy[0]
        dirt /= np.linalg.norm(dirt)
        dirn = np.array((-dirt[1], dirt[0]))
        text_position = 0.5 * (xy[0] + xy[1])
        text_rotation = np.arctan2(dirt[1], dirt[0])
    else:
        arc = Arc(plane.xc, plane.yc, plane.r, points[0], points[1])
        xy = arc.get_polygon()

        anglen = 0.5 * (arc.angle1 + arc.angle2)
        dirn = np.array((np.cos(anglen), np.sin(anglen)))
        text_position = np.array((plane.xc + plane.r * dirn[0],
                                  plane.yc + plane.r * dirn[1]))
        text_rotation = anglen - np.pi/2

    if linestyle:
        path = mpl.patches.Polygon(xy, closed=False, fill=False,
                                   linestyle=linestyle)
        ax.add_patch(path)

    if loc:
        # text
        text = '($'
        for m in miller:
            if m >= 0:
                text += str(m)
            else:
                text += r'\overline{' + str(-m) + r'}'
        text += '$)'

        va, ha = loc2align(loc)
        ha = 'center'               # ignore ha
        text_rotation = np.degrees(text_rotation)

        plot_text(text, text_position, ha, va, offset, text_rotation)


def plot_grid(frame, dtheta=None, dphi=None, linestyle=':', fig=None, ax=None):
    """
    Plot a grid of theta=const and phi=const lines.

    :param frame: Frame of plotting area (Wedge).
    :param dtheta: Increment between theta=const lines (deg).
        dtheta=None means automatic, dtheta=0 means no grid.
    :param dphi: Increment between phi=const lines (deg).
        dphi=None means automatic, dphi=0 means no grid.
    :param fig: Figure to be plotted to
    :param ax: Axes to be plotted to.
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    frame_polygon = frame.get_polygon()

    if dtheta != 0:
        x = frame_polygon[:,0]
        xmin = np.max(x)
        xmax = np.max(x)
        r = np.hypot(frame_polygon[:,0], frame_polygon[:,1])
        theta = np.degrees(inverse_stereographic_projection(r))
        theta_max = np.max(theta)

        if dtheta is None:
            if theta_max > 50:
                dtheta = 10.
            elif theta_max > 20:
                dtheta = 5.
            elif theta_max > 10:
                dtheta = 2.
            elif theta_max > 5:
                dtheta = 1.
            elif theta_max > 2:
                dtheta = 0.5
            elif theta_max > 1:
                dtheta = 0.2
            else:
                dtheta = 0.1

        thetas = np.arange(0., theta_max, dtheta)

        for theta in thetas:
            r = stereographic_projection(np.radians(theta))
            circle = Circle(0., 0., r)
            points = frame.intersect(circle)

            angle1 = np.arctan2(points[0][1], points[0][0])
            angle2 = np.arctan2(points[1][1], points[1][0])
            dangle = angle2 - angle1
            if -np.pi < dangle < 0 or dangle > np.pi:
                angle1, angle2 = angle2, angle1
            circle_polygon = circle.get_polygon(angle1, angle2)
            circle_patch = mpl.patches.Polygon(
                circle_polygon, closed=False, fill=False, linestyle=linestyle,linewidth =1)
            ax.add_patch(circle_patch)

            if dtheta < 1:
                label = str(theta)
            else:
                label = str(int(theta))
            label += r'$^\circ$'
            position = circle_polygon[0,:]
            if np.hypot(*position) < xmax:
                plot_text(label, position, ha='center', va='top', offset=0.5)

    if dphi != 0:
        phi = np.arctan2(frame_polygon[:,1], frame_polygon[:,0])
        phi_min = np.degrees(np.min(phi))
        phi_max = np.degrees(np.max(phi))

        if dphi is None:
            delta_phi = phi_max - phi_min
            if delta_phi > 30:
                dphi = 15.
            elif delta_phi > 20:
                dphi = 10.
            elif delta_phi > 10:
                dphi = 5.
            else:
                dphi = 1.

        imin = round(phi_min/dphi + 1)
        phi_min = imin * dphi
        imax = - round(- phi_max/dphi + 1)
        phi_max = imax * dphi
        phis = np.linspace(phi_min, phi_max, int(imax-imin+1))

        for phi in phis:
            line = Line(point1=(0,0),
                        point2=(np.cos(np.radians(phi)),
                                np.sin(np.radians(phi))))
            points = frame.intersect(line)
            if len(points) != 2:
                print(points)
                print('plot_grid: skipping phi=' + str(phi))
                continue
            x = (points[0][0], points[1][0])
            y = (points[0][1], points[1][1])
            ax.plot(x, y, linestyle=linestyle, color='k',linewidth =1)

            label = str(int(phi)) + r'$^\circ$'
            position = points[1]
            plot_text(label, position, ha='left', offset=0.5)


def plot_data(x, y, z, frame_patch, text='', ticks=None, fig=None, ax=None):
    """
    Plot the data clipped by frame_patch.

    The data is provided on a structured 2D grid.
    :param x: x coordinates (NXxNY array)
    :param y: y coordinates (NXxNY array)
    :param z: function values (NXxNY array)
    :param frame_patch: frame to clip to
    """
    if fig is None:
        fig = plt.gcf()
    if ax is None:
        ax = plt.gca()

    zmin = np.min(z)
    zmax = np.max(z)
    if ticks is None:
        extend = 'neither'
    else:
        if ticks[0] > zmin:
            if ticks[-1] < zmax:
                extend = 'both'
            else:
                extend = 'min'
        else:
            if ticks[-1] < zmax:
                extend = 'max'
            else:
                extend = 'neither'
        zmin = ticks[0]
        zmax = ticks[-1]

    cs = ax.contourf(x, y, z, np.linspace(zmin, zmax), cmap='jet',
                     extend=extend)
    cs.cmap.set_over('darkred')
    cs.cmap.set_under('darkblue')

    #clip data
    for c in cs.collections:
        c.set_clip_path(frame_patch)

    # colorbar
    cax = inset_axes(ax,
                     width="60%",  # width = 60% of parent_bbox width
                     height="5%",  # height = 5% of parent_bbox width
                     loc='upper left',
                     bbox_to_anchor=(0,0,1,1),
                     bbox_transform=ax.transAxes,
                     borderpad=0,
                     )

    cbar = plt.colorbar(cs, cax=cax, orientation='horizontal',
                        ticks=ticks)
    cbar.ax.set_xlabel(text, fontsize='x-large')
    cbar.ax.xaxis.set_label_position('top')


if __name__ == '__main__':
    fig, ax = plt.subplots()
    frame = Wedge()
    frame_polygon = frame.get_polygon()
    frame_patch = mpl.patches.Polygon(frame_polygon, closed=True, fill=False,
                                      color='k')

    ax.add_patch(frame_patch)

    plot_grid(frame)

    plot_direction((2,1,3), loc='center right')
    plot_direction((1,0,1), loc='bottom', marker=None)
    plot_direction((1,1,1), loc='center right', marker=None)
    plot_direction((0,0,1), loc='bottom', marker=None)

    plot_plane((1,-2,0), frame, '--', loc='top')
    plot_plane((1,1,-1), frame, '--', loc='bottom')


    ax.set_xlim(np.min(frame_polygon[:,0]), np.max(frame_polygon[:,0]))
    ax.set_ylim(np.min(frame_polygon[:,1]), np.max(frame_polygon[:,1]))
    ax.set_axis_off()
    ax.set_aspect('equal')
    plt.show()
