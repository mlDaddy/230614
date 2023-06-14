import open3d as o3d
import yaml
import os
import numpy as np
import random
from math import sin, cos


def load_yml(fpath):
    with open(fpath, 'r') as f:
        ff = yaml.load(f, Loader=yaml.FullLoader)
    return ff


def get_intrinsics(dataset_path, cam):
    fpath = os.path.join(dataset_path, 'calibration', 'intrinsics', cam + '_640x480.yml')
    f = load_yml(fpath)
    f = f['color']
    #f = f['depth']
    return f['fx'], f['fy'], f['ppx'], f['ppy']


def get_extrinsics(dataset_path, sub, seq, cam):
    metapath = os.path.join(dataset_path, sub, seq, 'meta.yml')
    metafile = load_yml(metapath)
    extrinsic_id = metafile['extrinsics']
    extrinsicpath = os.path.join(dataset_path, 'calibration', 'extrinsics_' + extrinsic_id, 'extrinsics.yml')
    extrinsicfile = load_yml(extrinsicpath)
    extrinsics = np.asarray(extrinsicfile['extrinsics'][cam])
    last_row = np.array([0., 0., 0., 1.])
    extrinsics = np.concatenate((extrinsics, last_row))
    extrinsics = extrinsics.reshape((4, 4))
    return extrinsics


def compile_pcd(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def rotation_augmentation(points_list):
    out = []
    theta = random.choice([120, -120])
    s = sin(theta)
    c = cos(theta)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rx @ Ry @ Rz
    for p in points_list:
        out.append((R @ p.T).T)
    return out


def perspective_correction(points_list, dataset_path, sub, seq, cam):
    out = []
    Tx = get_extrinsics(dataset_path, sub, seq, cam)
    for p in points_list:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p)
        pcd.transform(Tx)
        out.append(np.asarray(pcd.points))
    return out

def compile_pcd_pts(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)) * np.array(color))
    return pcd

def visualize_fwd_flows(points, points_fwd, flow_fwd):
    fwd_est = compile_pcd_pts(points + flow_fwd, [1, 0, 0])
    fwd = compile_pcd_pts(points_fwd, [0, 0, 1])
    pcd = compile_pcd_pts(points, [1, 1, 1])
    o3d.visualization.draw_geometries([pcd, fwd, fwd_est])
    o3d.visualization.draw_geometries([fwd, fwd_est])
    return

def visualize_fwd_flows_color(points, colors, points_fwd, colors_fwd, flow_fwd):
    fwd_est = compile_pcd(points + flow_fwd, colors)
    fwd = compile_pcd(points_fwd, colors_fwd)
    pcd = compile_pcd(points, colors)
    #o3d.visualization.draw_geometries([pcd, fwd, fwd_est])
    o3d.visualization.draw_geometries([fwd, fwd_est])
    #o3d.visualization.draw_geometries([fwd])
    #o3d.visualization.draw_geometries([fwd_est])
    return

def visualize_pts_list(plist):
    a = []
    for p in plist:
        pcd = compile_pcd_pts(p, [random.uniform(0, 1) for x in range(3)])
        a.append(pcd)
    o3d.visualization.draw_geometries(a)
    return
def visualize_fwd_flows(points, points_fwd, flow_fwd):
    fwd_est = compile_pcd_pts(points + flow_fwd, [1, 0, 0])
    fwd = compile_pcd_pts(points_fwd, [0, 0, 1])
    pcd = compile_pcd_pts(points, [0, 1, 0])
    #o3d.visualization.draw_geometries([pcd, fwd, fwd_est])
    o3d.visualization.draw_geometries([fwd, fwd_est])
    return
