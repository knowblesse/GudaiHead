
"""
Read the video and buttered csv data, check, and relabel if necessary
"""
import sys
from pathlib import Path
import cv2 as cv
from tkinter.filedialog import askdirectory
import numpy as np
import sys
from pathlib import Path

# Constants
TANK_PATH = Path(askdirectory())

# Find the path to the video
vidlist = []
vidlist.extend([i for i in TANK_PATH.glob('*.mkv')])
vidlist.extend([i for i in TANK_PATH.glob('*.avi')])
vidlist.extend([i for i in TANK_PATH.glob('*.mp4')])
vidlist.extend([i for i in TANK_PATH.glob('*.mpg')])
if len(vidlist) == 0:
    raise(BaseException(f'ReLabeler : Can not find video in {TANK_PATH}'))
elif len(vidlist) > 1:
    raise(BaseException(f'ReLabeler : Multiple video files found in {TANK_PATH}'))
else:
    path_video = vidlist[0]

# Find the csv to the video
if sorted(TANK_PATH.glob('*tracking.csv')):
    path_csv = next(TANK_PATH.glob('*tracking.csv'))
else:
    raise (BaseException(f'ReLabeler : Can not find tracking file in {TANK_PATH}'))

# Load the video and the label data
vid = cv.VideoCapture(str(path_video))
data = np.loadtxt(str(next(TANK_PATH.glob('*tracking.csv'))), delimiter=',')
num_frame = vid.get(cv.CAP_PROP_FRAME_COUNT)
fps = vid.get(cv.CAP_PROP_FPS)
lps = fps/data[1,0] # labels per second
current_label_index = 0

def calculatevelocity():
    # Find the excursion using velocity
    velocity_robot = ((data[1:,1] - data[0:-1,1]) ** 2 + (data[1:,2] - data[0:-1,2]) ** 2) ** 0.5
    velocity_robot = np.append(velocity_robot, 0) # To make equal size
    possibleExcursion_robot = np.abs(velocity_robot) > (np.mean(velocity_robot) + 3*np.std(velocity_robot))

    velocity_rat = ((data[1:,3] - data[0:-1,3]) ** 2 + (data[1:,4] - data[0:-1,4]) ** 2) ** 0.5
    velocity_rat = np.append(velocity_rat, 0) # To make equal size
    possibleExcursion_rat = np.abs(velocity_rat) > (np.mean(velocity_rat) + 3*np.std(velocity_rat))

    return (np.logical_or(possibleExcursion_robot, possibleExcursion_rat), possibleExcursion_robot, possibleExcursion_rat, velocity_robot, velocity_rat)


possibleExcursion, possibleExcursion_robot, possibleExcursion_rat, velocity_robot, velocity_rat = calculatevelocity()

# Main UI functions and callbacks
def getFrame(label_index, isShiftPressed):
    current_frame = int(data[label_index,0])
    vid.set(cv.CAP_PROP_POS_FRAMES, current_frame)
    ret, image = vid.read()
    if not ret:
        raise(BaseException('Can not read the frame'))
    cv.putText(image, f'{current_frame} - {label_index/data.shape[0]*100:.2f}% - Excursion {np.sum(possibleExcursion)} ({np.sum(possibleExcursion_robot)}, {np.sum(possibleExcursion_rat)})', [0,int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)-40)],fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=0.8, color=[255,255,255], thickness=2)
    cv.putText(image, 'a/f : +-10 label | s/d : +-1 label | e : goto rat excursion | w : goto robot excursion | g : input multiple position | q : quit', [0,30], fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1.2, color=[255,255,255], thickness=2)
    if data[label_index,1] != -1:
        cv.circle(image, (round(data[label_index,1]), round(data[label_index,2])), 15, [0,0,255], -1 ) # robot
        if isShiftPressed:
            cv.circle(image, (round(data[label_index,3]), round(data[label_index,4])), 15, [0,255,0], 3 ) # rat
        else:
            cv.circle(image, (round(data[label_index, 3]), round(data[label_index, 4])), 15, [0, 255, 0], -1)  # rat
    return image

class LabelObject:
    def initialize(self, image):
        self.image_org = image
        self.image = image
        self.rat_coordinate = []
        self.robot_coordinate = []
        self.isLabelUpdated_robot = False
        self.isLabelUpdated_rat = False

def drawLine(event, x, y, f, obj):
    if  event == cv.EVENT_LBUTTONUP:
        obj.isLabelUpdated_rat = True
        obj.rat_coordinate = [x, y]
    elif event == cv.EVENT_RBUTTONUP:
        obj.isLabelUpdated_robot = True
        obj.robot_coordinate = [x, y]

# Start Main UI
key = ''
labelObject = LabelObject()
cv.namedWindow('Main')
cv.setMouseCallback('Main', drawLine, labelObject)
labelObject.initialize(getFrame(current_label_index, False))
isShiftPressed = False
shiftPressedLabelIndex = 0
shiftReleasedLabelIndex = 0
shiftLabel = []

def refreshScreen(isShiftPressed):
    labelObject.initialize(getFrame(current_label_index, isShiftPressed))
    possibleExcursion[current_label_index] = False
    possibleExcursion_robot[current_label_index] = False
    possibleExcursion_rat[current_label_index] = False
    velocity_rat[current_label_index] = 0
    velocity_robot[current_label_index] = 0

# Last manual label
lastManualLabel = []
while key!=ord('q'):
    cv.imshow('Main', labelObject.image)
    key = cv.waitKey(1)
    if key == ord('a'): # backward 10 labels
        current_label_index = int(np.max([0, current_label_index - 10]))
        refreshScreen(isShiftPressed)
    elif key == ord('f'): # forward 10 labels
        current_label_index = int(np.min([data.shape[0]-1, current_label_index + 10]))
        refreshScreen(isShiftPressed)
    elif key == ord('s'): # backward 1 label
        current_label_index = int(np.max([0, current_label_index - 1]) )
        refreshScreen(isShiftPressed)
    elif key == ord('d'): # forward 1 label
        current_label_index = int(np.min([data.shape[0]-1, current_label_index + 1]))
        refreshScreen(isShiftPressed)
    elif key == ord('e'): # read the next possible excursion of rat
        foundExcursionIndex = np.argmax(np.abs(velocity_rat))
        current_label_index = foundExcursionIndex
        refreshScreen(isShiftPressed)
    elif key == ord('w'): # read the next possible excursion of robot
        foundExcursionIndex = np.argmax(np.abs(velocity_robot))
        current_label_index = foundExcursionIndex
        refreshScreen(isShiftPressed)
    elif key == ord('g'): # retain the current position until next label
        # Check if this function was activated.
        if isShiftPressed: # Cancel and initialize
            isShiftPressed = False
            shiftLabel = []
            shiftPressedLabelIndex = []
            shiftReleasedLabelIndex = []
            refreshScreen(False)
        else:
            isShiftPressed = True
            shiftPressedLabelIndex = current_label_index
            refreshScreen(True)

    elif key == ord(' '):
        # refresh velocity
        possibleExcursion, possibleExcursion_robot, possibleExcursion_rat, velocity_robot, velocity_rat = calculatevelocity()

    if labelObject.isLabelUpdated_robot or labelObject.isLabelUpdated_rat:
        if labelObject.isLabelUpdated_robot:
            data[current_label_index, 1:3] = labelObject.robot_coordinate
            labelObject.initialize(getFrame(current_label_index, isShiftPressed))
        if labelObject.isLabelUpdated_rat:
            if isShiftPressed: # change all labels from the shiftPressedLabelIndex
                data[np.min((shiftPressedLabelIndex, current_label_index)) : np.max((shiftPressedLabelIndex, current_label_index)) + 1, 3:5] = labelObject.rat_coordinate
                isShiftPressed = False
            else:
                data[current_label_index, 3:5] = labelObject.rat_coordinate
            labelObject.initialize(getFrame(current_label_index, isShiftPressed))

cv.destroyWindow('Main')
np.savetxt(str(path_csv), data,fmt='%d',delimiter=',')
