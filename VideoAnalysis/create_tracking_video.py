"""
Read the video and buttered csv data, Label each frame using interpolation, and save it into a video
"""
import cv2 as cv
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from pathlib import Path

# Constants
TANK_PATH = Path("E:\Data_fib\Robot Predator\Rtest3\R12")

# Find the path to the video
vidlist = []
vidlist.extend([i for i in TANK_PATH.glob('*.mkv')])
vidlist.extend([i for i in TANK_PATH.glob('*.avi')])
vidlist.extend([i for i in TANK_PATH.glob('*.mp4')])
if len(vidlist) == 0:
    raise(BaseException(f'SaveLabeledVideo : Can not find video in {TANK_PATH}'))
elif len(vidlist) > 1:
    raise(BaseException(f'SaveLabeledVideo : Multiple video files found in {TANK_PATH}'))
else:
    path_video = vidlist[0]

# Find the csv to the video
path_csv = path_video.parent / 'tracking.csv'

# Load the video and meta data
vid = cv.VideoCapture(str(path_video))

# get total number of frame
#   I can not trust vid.get(cv.CAP_PROP_FRAME_COUNT), because sometime I can't retrieve the last frame with vid.read()
num_frame = int(vid.get(cv.CAP_PROP_FRAME_COUNT))
vid.set(cv.CAP_PROP_POS_FRAMES, num_frame)
ret, _ = vid.read()
while not ret:
    print(f'SaveLabeledVideo : Can not read the frame from the last position. Decreasing the total frame count')
    num_frame -= 1
    vid.set(cv.CAP_PROP_POS_FRAMES, num_frame)
    ret, _ = vid.read()
fps = vid.get(cv.CAP_PROP_FPS)

# Load the label data
data = np.loadtxt(str(path_csv), delimiter=',')
lps = fps/data[1,0] # labels per second

robot_row_intp = interp1d(data[:, 0], data[:, 1], kind='linear')
robot_col_intp = interp1d(data[:, 0], data[:, 2], kind='linear')
rat_row_intp = interp1d(data[:, 0], data[:, 3], kind='linear')
rat_col_intp = interp1d(data[:, 0], data[:, 4], kind='linear')


# Save Video
vid_out = cv.VideoWriter(
    str(path_video.parent / (path_video.stem + '_labeled.' + path_video.suffix)),
    cv.VideoWriter_fourcc(*'MP4V'),
    fps,
    (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))),
    isColor=True)

vid = cv.VideoCapture(str(path_video))
for idx in tqdm(np.arange(num_frame-10)):
    idx = int(idx)
    ret, image = vid.read()
    if not ret:
        raise(BaseException('Can not read the frame'))

    cv.circle(image, (int(robot_row_intp(idx)), int(robot_col_intp(idx))), 30, [0,0,255], 3)
    cv.circle(image, (int(rat_row_intp(idx)), int(rat_col_intp(idx))), 30, [0,255,255], 3)
    vid_out.write(image)

vid_out.release()
