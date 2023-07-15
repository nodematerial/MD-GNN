import sys
import math
import yaml
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np


def file_reader(filepath, cutoff, division):
    with open(filepath)as f:
        for _ in range(3):
            next(f)
        num_atoms = int(f.readline().rstrip('\n'))
        for _ in range(5):
            next(f)
        mesh, xyz, feature = particles_info(f, num_atoms, division)
        neibcells = calc_neibcells(division)
        neibcoords = get_neibcoord(mesh, xyz, neibcells)
        edges, edge_per_node = make_graph(neibcoords, num_atoms, cutoff)
        feature = np.concatenate([feature, edge_per_node], 1)

    return edges, feature


def particles_info(f, num_atoms, division):
    mesh = defaultdict(set)
    xyz = defaultdict(list)
    feature = []
    for _ in range(num_atoms):
        s = f.readline()
        id, type, x, y, z, xs, ys, zs, vx, vy, vz = s.split()
        id, type = map(int, [id, type])
        id -= 1
        x, y, z, xs, ys, zs, vx, vy, vz = map(float, [x, y, z, xs, ys, zs, vx, vy, vz])
        mesh_id = mesh_classifier(xs, ys, zs, division)
        mesh[mesh_id].add(id)
        xyz[id] = [x, y, z]
        feature.append([x, y, z, vx, vy, vz, calc_speed(vx, vy, vz)])
    feature = np.array(feature)
    return mesh, xyz, feature


def mesh_classifier(xs, ys, zs, division):
    xdiv, ydiv, zdiv = division
    X = xdiv * xs // 1
    Y = ydiv * ys // 1
    Z = zdiv * zs // 1
    xyz = []
    for num, division in zip([X, Y, Z], [xdiv, ydiv, zdiv]):
        if num == division:
            xyz.append(division - 1)
        elif num == -1:
            xyz.append(0)
        else:
            xyz.append(num)
    return int(xyz[0] + xdiv * xyz[1] + (xdiv * ydiv) * xyz[2])


def calc_neibcells(division):
    xdiv, ydiv, zdiv = division

    # This variable represents the incremental increase in id for a shift of 1 in the y- and z-directions.
    xid_increment = 1
    yid_increment = xdiv
    zid_increment = xdiv * ydiv

    neibcells = []
    for i in range(xdiv):
        for j in range(ydiv):
            for k in range(zdiv):
                neibcell = []
                center_id = i + yid_increment * j + zid_increment * k
                neibcell.append(center_id)
                if i != 0:
                    neibcell.append(center_id - xid_increment)
                if i != (xdiv - 1):
                    neibcell.append(center_id + xid_increment)
                if j != 0:
                    neibcell.append(center_id - yid_increment)
                if j != (ydiv - 1):
                    neibcell.append(center_id + yid_increment)
                if k != 0:
                    neibcell.append(center_id - zid_increment)
                if k != (zdiv - 1):
                    neibcell.append(center_id + zid_increment)
                neibcells.append(neibcell)
    return neibcells


def get_neibcoord(mesh, xyz, neibcells):
    neibcoords = []
    for neibcell in neibcells:
        nodes = set()
        for mesh_id in neibcell:
            nodes = nodes.union(mesh[mesh_id])
        neibcoord = dict(filter(lambda item: item[0] in nodes, xyz.items()))
        assert len(neibcoord) == len(nodes)
        neibcoords.append(neibcoord)
    return neibcoords


def make_graph(neibcoords, num_atoms, cutoff):
    edges = []
    edge_per_node = np.zeros(num_atoms)
    for neibcoord in neibcoords:
        id_combs = combinations(neibcoord.keys(), 2)
        for comb in id_combs:
            i, j = comb[0], comb[1]
            A = neibcoord[i]
            B = neibcoord[j]
            distance = calc_distance(A, B)
            if distance < cutoff:
                edges.append((min(i, j), max(i, j)))
    edges = set(edges)
    for i, j in edges:
        edge_per_node[i] += 1
        edge_per_node[j] += 1
    print(f'number of edges: {len(edges)}')
    return np.array(list(edges)).transpose(), edge_per_node[:, np.newaxis]


def calc_distance(A, B):
    ax, ay, az = A
    bx, by, bz = B
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def calc_speed(vx, vy, vz):
    return math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)


def pipeline(cutoff, dirname, time, division):
    print(f'time: {time}')
    filepath = Path('dumpfiles') / dirname / f'{time}.alloysl'
    feature_path = Path('dataset') / dirname / f'{cutoff}/x/{time}'
    edges_path = Path('dataset') / dirname / f'{cutoff}/edges/{time}'
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.parent.mkdir(parents=True, exist_ok=True)

    edges, feature = file_reader(filepath, cutoff, division)
    np.save(feature_path, feature)
    np.save(edges_path, edges)


def main():
    with open(sys.argv[1], 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
        cutoff = CFG['cutoff']
        dirname = CFG['dirname']
        time = CFG['time']
        division = CFG['division']

    if type(time) is list:
        times = range(time[0], time[1] + 1, time[2])
        for t in times:
            pipeline(cutoff, dirname, t, division)
    else:
        pipeline(cutoff, dirname, time, division)


if __name__ == '__main__':
    main()
