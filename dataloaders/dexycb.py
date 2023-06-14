'''
no. of views = 1
truncated by trunc (optional)
'''

import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import open3d as o3d
from os.path import join, split
from dataloaders.global_vars import *
from dataloaders.utils import *


class multi_view_dataset(Dataset):
    def __init__(self,
                 args=None,
                 dataset_path=DEX_YCB_DIR,
                 csv_file='dataloaders/dataset_files/validate_dataset.csv',
                 npoints=2048,
                 trunc=None,
                 REV_FLOW=False,
                 AUGMENTATION=False,
                 HANDS_ONLY=False,
                 serial_list=['836212060125']):
        self.csv = pd.read_csv(csv_file)
        self.dataset_path = dataset_path
        self.npoints = npoints
        self.args = args
        self.trunc = trunc
        self.REV_FLOW = REV_FLOW
        self.AUGMENTATION = AUGMENTATION
        self.HANDS_ONLY = HANDS_ONLY
        self.serial_list = serial_list

    def __len__(self):
        if self.trunc is None:
            return len(self.csv['num_frames'])
        else:
            return self.trunc

    def __getitem__(self, idx):
        if self.REV_FLOW == True:
            refid, mano_points_1, points_1, colors_1, points_2, colors_2, flow_fwd, mano_flow_fwd, points_0, colors_0, flow_rev, mano_flow_rev = self.fetch_data(
                idx)
            if self.AUGMENTATION == True:
                mano_points_1, points_1, points_2, flow_fwd, mano_flow_fwd, points_0, flow_rev, mano_flow_rev = rotation_augmentation(
                    [mano_points_1, points_1, points_2, flow_fwd, mano_flow_fwd, points_0, flow_rev, mano_flow_rev])
            return refid, mano_points_1, points_1, colors_1, points_2, colors_2, flow_fwd, mano_flow_fwd, points_0, colors_0, flow_rev, mano_flow_rev
        else:
            refid, mano_points_1, points_1, colors_1, points_2, colors_2, flow_fwd, mano_flow_fwd = self.fetch_data(idx)
            mano_points_1, points_1, points_2, flow_fwd, mano_flow_fwd = rotation_augmentation(
                [mano_points_1, points_1, points_2, flow_fwd, mano_flow_fwd])
            return refid, mano_points_1, points_1, colors_1, points_2, colors_2, flow_fwd, mano_flow_fwd

    def fetch_data(self, idx):

        subject, sequence, frame = self.csv.iloc[idx]
        refid = f"{subject}, sequence={sequence}, frame={frame}"

        frame_dir_list = [join(self.dataset_path,
                         PROCESSED_DATA_SUB_DIR,
                         subject,
                         sequence,
                         x)
                     for x in self.serial_list]

        mano_points_0_combined = []
        mano_points_1_combined = []
        mano_points_2_combined = []
        points_1_combined = []
        colors_1_combined = []
        points_2_combined = []
        colors_2_combined = []
        flow_fwd_combined = []
        points_0_combined = []
        colors_0_combined = []
        flow_rev_combined = []
        mano_points_0_counter = 0
        mano_points_1_counter = 0
        mano_points_2_counter = 0
        a = []

        for i, frame_dir in enumerate(frame_dir_list, 0):
            frame1_npz = np.load(join(frame_dir, pcd_name_format.format(frame)))
            frame2_npz = np.load(join(frame_dir, pcd_name_format.format(frame + 1)))

            mano_points_1 = frame1_npz['mano_points']
            hand_exists_1 = frame1_npz['hand_exists']
            if hand_exists_1:
                mano_points_1_counter += 1
                [mano_points_1] = perspective_correction([mano_points_1],
                                                         self.dataset_path, subject, sequence, self.serial_list[i]
                                                         )
            mano_points_2 = frame2_npz['mano_points']
            hand_exists_2 = frame2_npz['hand_exists']
            if hand_exists_2:
                mano_points_2_counter += 1
                [mano_points_2] = perspective_correction([mano_points_2],
                                                         self.dataset_path, subject, sequence, self.serial_list[i]
                                                         )

            if self.HANDS_ONLY:
                pcd_1 = compile_pcd(frame1_npz['hand_pcd_points'], frame1_npz['hand_pcd_colors'])
                pcd_2 = compile_pcd(frame2_npz['hand_pcd_points'], frame2_npz['hand_pcd_colors'])
                flow_fwd =frame1_npz['hand_flow_fwd']
            else:
                pcd_1 = compile_pcd(np.concatenate((frame1_npz['hand_pcd_points'], frame1_npz['ycb_pcd_points'])),
                                    np.concatenate((frame1_npz['hand_pcd_colors'], frame1_npz['ycb_pcd_colors'])))
                pcd_2 = compile_pcd(np.concatenate((frame2_npz['hand_pcd_points'], frame2_npz['ycb_pcd_points'])),
                                    np.concatenate((frame2_npz['hand_pcd_colors'], frame2_npz['ycb_pcd_colors'])))
                flow_fwd = np.concatenate((frame1_npz['hand_flow_fwd'], frame1_npz['ycb_flow_fwd']))

            points_1 = np.asarray(pcd_1.points)
            colors_1 = np.asarray(pcd_1.colors)
            points_2 = np.asarray(pcd_2.points)
            colors_2 = np.asarray(pcd_2.colors)

            temp = points_1 + flow_fwd
            [points_1, points_2, temp] = perspective_correction(
                [points_1, points_2, temp],
                self.dataset_path, subject, sequence, self.serial_list[i]
            )
            flow_fwd = temp - points_1

            if i == 0:
                mano_points_1_combined = mano_points_1
                points_1_combined = points_1
                colors_1_combined = colors_1
                mano_points_2_combined = mano_points_2
                points_2_combined = points_2
                colors_2_combined = colors_2
                flow_fwd_combined = flow_fwd
            else:
                mano_points_1_combined += mano_points_1
                points_1_combined = np.concatenate((points_1_combined, points_1))
                colors_1_combined = np.concatenate((colors_1_combined, colors_1))
                mano_points_2_combined += mano_points_2
                points_2_combined = np.concatenate((points_2_combined, points_2))
                colors_2_combined = np.concatenate((colors_2_combined, colors_2))
                flow_fwd_combined = np.concatenate((flow_fwd_combined, flow_fwd))

            if self.REV_FLOW:
                frame0_npz = np.load(join(frame_dir, pcd_name_format.format(frame - 1)))

                mano_points_0 = frame0_npz['mano_points']
                hand_exists_0 = frame0_npz['hand_exists']
                if hand_exists_0:
                    mano_points_0_counter += 1
                    [mano_points_0] = perspective_correction([mano_points_0],
                                                             self.dataset_path, subject, sequence, self.serial_list[i]
                                                             )

                pcd_0 = compile_pcd(np.concatenate((frame0_npz['hand_pcd_points'], frame0_npz['ycb_pcd_points'])),
                                    np.concatenate((frame0_npz['hand_pcd_colors'], frame0_npz['ycb_pcd_colors'])))
                flow_rev = np.concatenate((frame1_npz['hand_flow_rev'], frame1_npz['ycb_flow_rev']))

                points_0 = np.asarray(pcd_0.points)
                colors_0 = np.asarray(pcd_0.colors)

                temp = points_1 + flow_rev
                [points_0, temp] = perspective_correction(
                    [points_0, flow_rev],
                    self.dataset_path, subject, sequence, self.serial_list[i]
                )
                flow_rev = temp - points_1

                if i == 0:
                    mano_points_0_combined = mano_points_0
                    points_0_combined = points_0
                    colors_0_combined = colors_0
                    flow_rev_combined = flow_rev

                else:
                    mano_points_0_combined += mano_points_0
                    points_0_combined = np.concatenate((points_0_combined, points_0))
                    colors_0_combined = np.concatenate((colors_0_combined, colors_0))
                    flow_rev_combined = np.concatenate((flow_rev_combined, flow_rev))

        if mano_points_1_counter > 0:
            mano_points_1_combined /= mano_points_1_counter
        if mano_points_2_counter > 0:
            mano_points_2_combined /= mano_points_2_counter

        if np.sum(mano_points_1_combined * mano_points_2_combined) > 0:
            mano_flow_fwd_combined = mano_points_2_combined - mano_points_1_combined
        else:
            mano_flow_fwd_combined = np.zeros((800, 3))

        if np.sum(points_1_combined) == 0 or np.sum(points_2_combined) == 0:
            points_1_combined = np.zeros((self.npoints, 3))
            points_2_combined = np.zeros((self.npoints, 3))
            colors_1_combined = np.zeros((self.npoints, 3))
            colors_2_combined = np.zeros((self.npoints, 3))
            flow_fwd_combined = np.zeros((self.npoints, 3))

        selection_1 = np.random.choice(points_1_combined.shape[0], self.npoints)
        points_1_combined = points_1_combined[selection_1, :]
        colors_1_combined = colors_1_combined[selection_1, :]
        flow_fwd_combined = flow_fwd_combined[selection_1, :]

        selection_2 = np.random.choice(points_2_combined.shape[0], self.npoints)
        points_2_combined = points_2_combined[selection_2, :]
        colors_2_combined = colors_2_combined[selection_2, :]

        shuffle_idx_1 = np.arange(self.npoints)
        np.random.shuffle(shuffle_idx_1)
        points_1_combined = points_1_combined[shuffle_idx_1]
        colors_1_combined = colors_1_combined[shuffle_idx_1]
        flow_fwd_combined = flow_fwd_combined[shuffle_idx_1]

        shuffle_idx_2 = np.arange(self.npoints)
        np.random.shuffle(shuffle_idx_2)
        points_2_combined = points_2_combined[shuffle_idx_2]
        colors_2_combined = colors_2_combined[shuffle_idx_2]

        out = [refid, mano_points_1_combined.astype(np.float32),
               points_1_combined.astype(np.float32), colors_1_combined.astype(np.float32),
               points_2_combined.astype(np.float32), colors_2_combined.astype(np.float32),
               flow_fwd_combined.astype(np.float32), mano_flow_fwd_combined.astype(np.float32)]

        if self.REV_FLOW:

            if mano_points_0_counter > 0:
                mano_points_0_combined /= mano_points_0_counter

            if np.sum(mano_points_1_combined * mano_points_0_combined) > 0:
                mano_flow_rev_combined = mano_points_0_combined - mano_points_1_combined
            else:
                mano_flow_rev_combined = np.zeros((800, 3))

            flow_rev_combined = flow_rev_combined[selection_1, :]

            selection_0 = np.random.choice(points_0_combined.shape[0], self.npoints)
            points_0_combined = points_0_combined[selection_0, :]
            colors_0_combined = colors_0_combined[selection_0, :]

            flow_rev_combined = flow_rev_combined[shuffle_idx_1]

            np.random.shuffle(shuffle_idx_2)
            points_0_combined = points_0_combined[shuffle_idx_2]
            colors_0_combined = colors_0_combined[shuffle_idx_2]

            out_rev = [points_0_combined.astype(np.float32), colors_0_combined.astype(np.float32),
                       flow_rev_combined.astype(np.float32), mano_flow_rev_combined.astype(np.float32)]
            out = out + out_rev

        return out

