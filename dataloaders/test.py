import sys
sys.path.append('//AA_PG_SEECS')
from dexycb import *
from utils import *
import open3d as o3d

multi_view_test = True
REV_FLOW = False

if multi_view_test:
    dataset = multi_view_dataset(csv_file='dataloaders/dataset_files/validate_dataset.csv',
                                 HANDS_ONLY=True,
                                 trunc=347,
                                 serial_list=['836212060125', '839512060362'])
else:
    dataset = multi_view_dataset(csv_file='dataloaders/dataset_files/validate_dataset.csv',
                                 trunc=347,
                                 serial_list=['836212060125'])

for i, data in enumerate(iter(dataset)):
    if REV_FLOW:
        refid, mano_points_1, points_1, colors_1, points_2, colors_2, flow_fwd, mano_flow_fwd, points_0, colors_0, flow_rev, mano_flow_rev = data
    else:
        refid, mano_points_1, points_1, colors_1, points_2, colors_2, flow_fwd, mano_flow_fwd = data
    print(i, end='\r')
    if i == 21:
        visualize_fwd_flows(points_1, points_2, flow_fwd)
        visualize_pts_list([mano_points_1, points_1])
        visualize_pts_list([mano_points_1 + mano_flow_fwd, points_2])
        visualize_pts_list([mano_points_1,
                            mano_points_1 + mano_flow_fwd,
                            points_1,
                            points_1 + flow_fwd,
                            points_2])
        if REV_FLOW:
            visualize_fwd_flows(points_1, points_2, flow_rev)
            visualize_pts_list([mano_points_1, points_1])
            visualize_pts_list([mano_points_1 + mano_flow_rev, points_2])
            visualize_pts_list([mano_points_1,
                                mano_points_1 + mano_flow_rev,
                                points_1,
                                points_1 + flow_rev,
                                points_2])
        print(refid)
        print(mano_points_1.shape, points_1.shape, points_2.shape, flow_fwd.shape)
        print(mano_points_1[0, :])
        exit()
