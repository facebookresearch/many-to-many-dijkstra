"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

"""
The pathfinder algorithm.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement
import heapq
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numba import autojit
import numpy as np


def seek(
    origins,
    targets=None,
    weights=None,
    path_handling='link',
    debug=False,
    film=False,
):
    """
    Find the shortest paths between *any* origin and *each* target.

    Pathfinder is a modified version of Dijkstra's algorithm
    (https://fburl.com/is86jvbn) for finding
    the shortest distance between two points in a graph. It differs in
    a few important ways:

    * It finds the shortest distance between a target point and the
        nearest of a set of origin points. This is then repeated for
        each target point.
    * It assumes a gridded topology. In other words, it knows that
        each node only
        touches its neighbors to the north, south, east, west,
        northeast, northwest, southeast, and southwest.

    Like Dijkstra's, Pathfinder assumes that all weights are 0 or greater.
    Negative weights are set to zero.  All input arrays (origins,
    targets, and weights) need to have the same number of rows and columns.

    @param origins, targets: 2D numpy array of ints
        Any non-zero values are the locations of origin and target points,
        respectively. Note that both may be modified. Target points may
        be removed, once paths to them are found and origins may be
        augmented (see path_handling param below).
        If targets is not supplied, no targets are assumed and a targets
        array of all zeros is created. This is when calculating minimum
        distances from a set of origins to all points of the grid.
    @param weights: 2D numpy array of floats
        The cost of visiting a grid square, zero or greater.
        For favorable (easy to traverse) grid locations, this is low.
        For unfavorable grid locations, this is high.
        If not supplied, a weights array of all ones is used. This is
        useful for calculating as-the-crow-flies distance.
    @param path_handling: string
        One of {'link', 'assimilate', 'none', 'l', 'a', 'n'}.
        Determines how to handle paths between target and origins,
        once they are found.

        * 'link' or 'l'  adds the target to the origins, as well as the path
            connecting them
        * 'assimilate' or 'a' adds the target to the origins, but does not
            add the path.
        * 'none' or 'n' doesn't add anything to the origins.

        These options allow for a variety of use cases.
        'link' is good for growing a network by connecting nodes,
        as in planning an electrical grid.
        'assimilate' is good for growing a network by adding nodes that
        have no physical connection between them, as in planning an
        ad-hoc wireless network.
        'none' is good for finding a path from a backbone or trunk
        to many leaf nodes, as in planning fiber backhaul routing.
    @param debug: boolean
        If True, provide text updates on the algorithm's progress.
    @param film: boolean
        If True, periodically save snapshots of the algorithm's progress.

    @retun results: dict
        'paths': 2D numpy array of ints
            One, where paths have been found, and zero everywhere else.
        'distance: 2D numpy array of floats
            The length of the shortest path (the sum of the weights of grid
            cells traversed) from the nearest origin point to every point
            on the grid. Origin points have a distance of zero and it
            goes up from there the further away you get.
        'rendering': 2D numpy array of floats
            An image representing the final state of the algorithm, including
            paths found and distances calculated.
    """
    if weights is None:
        weights = np.ones(origins.shape)
    if targets is None:
        targets = np.zeros(origins.shape, dtype=np.int8)
    assert targets.shape == origins.shape
    assert targets.shape == weights.shape
    path_handling = path_handling.lower()
    assert path_handling in ['none', 'n', 'assimilate', 'a', 'link', 'l']
    n_rows, n_cols = origins.shape
    if path_handling[0] == 'n':
        path_handling = 0
    if path_handling[0] == 'a':
        path_handling = 1
    if path_handling[0] == 'l':
        path_handling = 2

    iteration = 0
    not_visited = 9999999999.

    if film:
        frame_rate = int(1e4)
        frame_counter = 100000
        frame_dirname = 'frames'
        try:
            os.mkdir(frame_dirname)
        except Exception:
            # NBD
            pass

        cwd = os.getcwd()
        try:
            os.chdir(frame_dirname)
            for filename in os.listdir('.'):
                os.remove(filename)
        except Exception:
            print('Frame deletion failed')
        finally:
            os.chdir(cwd)

    rendering = 1. / (2. * weights)
    rendering = np.minimum(rendering, 1.)
    target_locations = np.where(targets)
    n_targets = target_locations[0].size
    n_targets_remaining = n_targets
    n_targets_remaining_update = n_targets
    for i_target, row in enumerate(target_locations[0]):
        col = target_locations[1][i_target]
        wid = 8
        rendering[
            row - wid:
            row + wid + 1,
            col - wid:
            col + wid + 1] = .5

    # The distance array shows the shortest weighted distance from
    # each point in the grid to the nearest origin point.
    distance = np.ones((n_rows, n_cols)) * not_visited
    origin_locations = np.where(origins != 0)
    distance[origin_locations] = 0.

    # The paths array shows each of the paths that are discovered
    # from targets to their nearest origin point.
    paths = np.zeros((n_rows, n_cols), dtype=np.int8)

    # The halo is the set of points under evaluation. They surround
    # the origin points and expand outward, forming a growing halo
    # around the set of origins that eventually enevlops targets.
    # It is implemented using a heap queue, so that the halo point
    # nearest to an origin is always the next one that gets evaluated.
    halo = []
    for i, origin_row in enumerate(origin_locations[0]):
        origin_col = origin_locations[1][i]
        heapq.heappush(halo, (0., (origin_row, origin_col)))

    # The temporary array for tracking locations to add to the halo.
    # This gets overwritten with each iteration.
    new_locs = np.zeros((int(1e6), 3))
    n_new_locs = 0

    while len(halo) > 0:
        iteration += 1
        if debug:
            if (n_targets_remaining > n_targets_remaining_update or
                    iteration % 1e4 == 0.):
                n_targets_remaining = n_targets_remaining_update
                print('\r {num} targets of {total} reached, {rem} remaining, {halo_len} to try '
                      .format(
                          num=n_targets - n_targets_remaining,
                          total=n_targets,
                          rem=n_targets_remaining,
                          halo_len=len(halo),
                      ), end='')
                sys.stdout.flush()
        if film:
            if iteration % frame_rate == 0:
                frame_counter = render(
                    distance,
                    frame_counter,
                    frame_dirname,
                    not_visited,
                    rendering,
                )

        # Reinitialize locations to add.
        new_locs[:n_new_locs, :] = 0.
        n_new_locs = 0

        # Retrieve and check the location with shortest distance.
        (distance_here, (row_here, col_here)) = heapq.heappop(halo)
        n_new_locs, n_targets_remaining_update = nb_loop(
            col_here,
            distance,
            distance_here,
            n_cols,
            n_new_locs,
            n_rows,
            n_targets_remaining,
            new_locs,
            not_visited,
            origins,
            path_handling,
            paths,
            row_here,
            targets,
            weights,
        )
        for i_loc in range(n_new_locs):
            loc = (int(new_locs[i_loc, 1]), int(new_locs[i_loc, 2]))
            heapq.heappush(halo, (new_locs[i_loc, 0], loc))

    if debug:
        print('\r                                                 ', end='')
        sys.stdout.flush()
        print('')
    # Add the newfound paths to the visualization.
    rendering = 1. / (1. + distance / 10.)
    rendering[np.where(origins)] = 1.
    rendering[np.where(paths)] = .8
    results = {'paths': paths, 'distance': distance, 'rendering': rendering}
    return results


def render(
    distance,
    frame_counter,
    frame_dirname,
    not_visited,
    rendering,
):
    """
    Turn the progress of the algorithm into a pretty picture.
    """
    progress = rendering.copy()
    visited_locs = np.where(distance < not_visited)
    progress[visited_locs] = 1. / (1. + distance[visited_locs] / 10.)
    filename = 'pathfinder_frame_' + str(frame_counter) + '.png'
    cmap = 'inferno'
    dpi = 1200
    plt.figure(33374)
    plt.clf()
    plt.imshow(
        progress,
        origin='higher',
        interpolation='nearest',
        cmap=plt.get_cmap(cmap),
        vmax=1.,
        vmin=0.,
    )
    filename_full = os.path.join(frame_dirname, filename)
    plt.savefig(filename_full, dpi=dpi)
    frame_counter += 1
    return frame_counter


@autojit(nopython=True)
def nb_trace_back(
    distance,
    n_new_locs,
    new_locs,
    not_visited,
    origins,
    path_handling,
    paths,
    target,
    weights,
):
    """
    Connect each found electrified target to the grid through
    the shortest available path.
    """
    # Handle the case where you find more than one target.
    path = []
    distance_remaining = distance[target]
    current_location = target
    while distance_remaining > 0.:
        path.append(current_location)
        (row_here, col_here) = current_location
        # Check each of the neighbors for the lowest distance to grid.
        neighbors = [
            ((row_here - 1, col_here), 1.),
            ((row_here + 1, col_here), 1.),
            ((row_here, col_here + 1), 1.),
            ((row_here, col_here - 1), 1.),
            ((row_here - 1, col_here - 1), 2.**.5),
            ((row_here + 1, col_here - 1), 2.**.5),
            ((row_here - 1, col_here + 1), 2.**.5),
            ((row_here + 1, col_here + 1), 2.**.5),
        ]
        lowest_distance = not_visited
        # It's confusing, but keep in mind that
        # distance[neighbor] is the distance from the neighbor position
        # to the grid, while neighbor_distance is
        # the distance *through*
        # the neighbor position to the grid. It is distance[neighbor]
        # plus the distance to the neighbor from the current position.
        for (neighbor, scale) in neighbors:
            if neighbor not in path:
                distance_from_neighbor = scale * weights[current_location]
                neighbor_distance = (distance[neighbor] +
                                     distance_from_neighbor)
                if neighbor_distance < lowest_distance:
                    lowest_distance = neighbor_distance
                    best_neighbor = neighbor

        # This will fail if caught in a local minimum.
        if distance_remaining < distance[best_neighbor]:
            distance_remaining = 0.
            continue

        distance_remaining = distance[best_neighbor]
        current_location = best_neighbor

    # Add this new path.
    for i_loc, loc in enumerate(path):
        paths[loc] = 1
        # If paths are to be linked, include the entire paths as origins and
        # add them to new_locs. If targets are to be assimilated, just add
        # the target (the first point on the path) to origins and new_locs.
        if path_handling == 2 or (
                path_handling == 1 and i_loc == 0):
            origins[loc] = 1
            distance[loc] = 0.
            new_locs[n_new_locs, 0] = 0.
            new_locs[n_new_locs, 1] = loc[0]
            new_locs[n_new_locs, 2] = loc[1]
            n_new_locs += 1

    return n_new_locs


@autojit(nopython=True)
def nb_loop(
    col_here,
    distance,
    distance_here,
    n_cols,
    n_new_locs,
    n_rows,
    n_targets_remaining,
    new_locs,
    not_visited,
    origins,
    path_handling,
    paths,
    row_here,
    targets,
    weights,
):
    """

    """
    # Calculate the distance for each of the 8 neighbors.
    neighbors = [
        ((row_here - 1, col_here), 1.),
        ((row_here + 1, col_here), 1.),
        ((row_here, col_here + 1), 1.),
        ((row_here, col_here - 1), 1.),
        ((row_here - 1, col_here - 1), 2.**.5),
        ((row_here + 1, col_here - 1), 2.**.5),
        ((row_here - 1, col_here + 1), 2.**.5),
        ((row_here + 1, col_here + 1), 2.**.5),
    ]

    for (neighbor, scale) in neighbors:
        weight = scale * weights[neighbor]
        neighbor_distance = distance_here + weight

        if distance[neighbor] == not_visited:
            if targets[neighbor]:
                n_new_locs = nb_trace_back(
                    distance,
                    n_new_locs,
                    new_locs,
                    not_visited,
                    origins,
                    path_handling,
                    paths,
                    neighbor,
                    weights,
                )
                targets[neighbor] = 0
                n_targets_remaining -= 1
        if neighbor_distance < distance[neighbor]:
            distance[neighbor] = neighbor_distance
            if (neighbor[0] > 0 and
                    neighbor[0] < n_rows - 1 and
                    neighbor[1] > 0 and
                    neighbor[1] < n_cols - 1):
                new_locs[n_new_locs, 0] = distance[neighbor]
                new_locs[n_new_locs, 1] = neighbor[0]
                new_locs[n_new_locs, 2] = neighbor[1]
                n_new_locs += 1
    return n_new_locs, n_targets_remaining
