import os

#DEX_YCB_DIR = os.environ['DEX_YCB_DIR']
DEX_YCB_DIR = '/home/adnan/ws/DEXYCB Dataset'
PROCESSED_DATA_SUB_DIR = 'processed' #subdirectory of DEX_YCB_DIR

color_name_format = "color_{:06d}.jpg"
depth_name_format = "aligned_depth_to_color_{:06d}.png"
label_name_format = "labels_{:06d}.npz"
sf_name_format = "sf_{:06d}.npy"
pcd_name_format = "pcd_{:06d}.npz"

SUBJECTS = [
    '20200709-subject-01',
    '20200813-subject-02',
    '20200820-subject-03',
    '20200903-subject-04',
    '20200908-subject-05',
    '20200918-subject-06',
    '20200928-subject-07',
    '20201002-subject-08',
    '20201015-subject-09',
    '20201022-subject-10',
]

SERIALS = [
    '840412060917',
    '836212060125',
    '839512060362',
    '841412060263',
    '932122060857',
    '932122060861',
    '932122061900',
    '932122062010',
]