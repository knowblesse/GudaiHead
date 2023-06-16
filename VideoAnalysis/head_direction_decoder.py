import queue
from queue import Queue
from threading import Thread
from tkinter import W
import cv2 as cv
import numpy as np
from pathlib import Path
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
import time
import tensorflow as tf
from tensorflow import keras


"""
1. Read frame
2. Extract Rat using human-corrected coordinate
3. get through head detection net
"""

def vector2degree(r1,c1,r2,c2):
    """
    calculate the degree of the vector.
    The vector is starting from (r1, c1) to (r2, c2).
    Beware that the coordinate is not x, y rather it is row, column.
    This row, column coordinate correspond to (inverted y, x). So, the 90 degree is the arrow going
    downward, and the 180 degree is the arrow going to the left.
    """
    # diagonal line
    l = ((r1 - r2) ** 2 + (c1 - c2) ** 2) ** 0.5
    # temporal degree value
    temp_deg = np.rad2deg(np.arccos((c2 - c1) / l))
    # if r1 <= r2, then [0, 180) degree = temp_deg
    # if r1 > r2, then [180. 360) degree = 360 - temp_deg
    deg = 360 * np.array(r1 > r2, dtype=int) + (np.array(r1 <= r2, dtype=int) - np.array(r1 > r2, dtype=int)) * temp_deg
    return np.round(deg).astype(int)

class AfterwardButter():
    def __init__(self, video_path=[]):
        # Get Video Path
        if video_path:
            self.video_path = video_path
        else:
            self.video_path = Path(askopenfilename())
            #self.video_path = Path(r"D:\Data_fib\Robot Predator\Rtest1\R1\2023-05-04 14-02-08_r1.mkv")

        self.vc = cv.VideoCapture(str(self.video_path.absolute()))

        # Get Video infor
        self.num_frame = self.vc.get(cv.CAP_PROP_FRAME_COUNT)
        self.fps = self.vc.get(cv.CAP_PROP_FPS)
        self.frame_height = int(self.vc.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.vc.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_size = (self.frame_height, self.frame_width)

        # Get tracking data
        try:
            self.tracking_data = np.loadtxt(next(self.video_path.parent.glob("tracking.csv")), delimiter=',', dtype=int)
        except StopIteration:
            print("Can not find tracking.csv")

        # Load Model
        self.predictBatchSize = 100
        model_path = r'C:\VCF\butter\Models\butterNet_V2'
        try:
            model = keras.models.load_model(str(model_path))
        except:
            raise(BaseException('VideoProcessor : Can not load model from ' + str(model_path)))
        self.model = model
        self.ROI_size = model.layers[0].input.shape[1]


    def save(self):
        np.savetxt((self.video_path.parent / 'head_degree.csv').absolute(), self.output_data, '%d', delimiter=',')

    def run(self):
        self.output_data = np.zeros((self.tracking_data.shape[0], 4), dtype=int)

        predictList = []
        indexList = []

        self.startROIextractionThread(self.tracking_data[:,0])
        frameList = tqdm(self.tracking_data[:,0])
        for idx, frameNumber in enumerate(frameList):
            if len(predictList) < self.predictBatchSize: # If batch not full, add more
                predictList.append(self.frameQ.get())
                indexList.append(idx)
            else: # if batch full, feed model
                testing = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(predictList))
                result = self.model.predict(testing)

                self.output_data[indexList, :] = np.concatenate((
                    self.tracking_data[indexList,:1],
                    (self.tracking_data[indexList, 3:5] + result[:, [1,0]] - int(self.ROI_size / 2)).astype(int),
                    np.expand_dims(vector2degree(result[:, 0], result[:, 1],result[:, 2], result[:, 3]), 1)
                ), axis=1)

                predictList = []
                indexList = []

        # Run for the last batch
        testing = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(predictList))
        result = self.model.predict(testing)

        self.output_data[indexList, :] = np.concatenate((
            self.tracking_data[indexList, :1],
            (self.tracking_data[indexList, 3:5] + result[:, [1, 0]] - int(self.ROI_size / 2)).astype(int),
            np.expand_dims(vector2degree(result[:, 0], result[:, 1], result[:, 2], result[:, 3]), 1)
        ), axis=1)

        self.isProcessed = True

    def startROIextractionThread(self, frameList):
        # Video IO Thread and Queue
        self.frameQ = Queue(maxsize=200)
        self.vcIOthread = Thread(target=self.__readVideo, args=(frameList,))
        self.vcIOthread.daemon = True # indicate helper thread
        if not self.vcIOthread.is_alive():
            self.vcIOthread.start()

        self.isMultithreading = True

    def __readVideo(self, frameList):
        """
        __readVideo : multithreading. read video, extract frame and store in self.frameQ
        """
        print('Video IO Thread started\n')

        currHeader = 0
        for idx, frameNumber in enumerate(frameList):
            while currHeader < frameNumber:
                ret = self.vc.grab()
                currHeader += 1
            ret, frame = self.vc.read()
            currHeader += 1

            blob_center_row = self.tracking_data[idx, 4]
            blob_center_col = self.tracking_data[idx, 3]

            self.half_ROI_size = int(self.ROI_size/2)

            expanded_image = cv.copyMakeBorder(frame, self.half_ROI_size, self.half_ROI_size, self.half_ROI_size,
                                               self.half_ROI_size,
                                               cv.BORDER_CONSTANT, value=[0, 0, 0])

            chosen_image = expanded_image[
                           blob_center_row - self.half_ROI_size + self.half_ROI_size: blob_center_row + self.half_ROI_size + self.half_ROI_size,
                           blob_center_col - self.half_ROI_size + self.half_ROI_size: blob_center_col + self.half_ROI_size + self.half_ROI_size,
                           :]


            self.frameQ.put(chosen_image)

        self.readVideoComplete = True
        print('Video IO Thread stopped\n')
        return

base_path = Path(r"D:\Data_fib\Robot Predator\Rtest1")

for idx, video_path in enumerate(base_path.rglob('*.mkv')):
    rrr = AfterwardButter(video_path)
    rrr.run()
    rrr.save()
    print(f'[{idx}]\n')
