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
import teleknock
from scipy.stats import norm

def paintROI(image):
    """
    left paint : Select
    right paint : Deselect
    mouse wheel : change brush size
    """
    image_orig = image.copy()

    cv.namedWindow('Paint ROI')
    cv.setMouseCallback('Paint ROI', func, obj)

    def mouseCallBack(event, x, y, f, obj):
        if event == cv.EVENT_LBUTTONDOWN:
        elif event == cv.EVENT_LBUTTONUP:
        elif event == cv.EVENT_RBUTTONDOWN:
        elif event == cv.EVENT_RBUTTONUP:
        else event == EVENT_MOUSEWHEEL 



    zoom_position = cv.selectROI('Select Robot', image)
    cv.destroyWindow('Select Robot')





class RobotRatRolling():
    def __init__(self, path=[]):
        # Get Video Path
        if path:
            self.video_path = path
        else:
            self.video_path = Path(askopenfilename())
        self.vc = cv.VideoCapture(str(self.video_path.absolute()))

        # Get Video infor
        self.num_frame = self.vc.get(cv.CAP_PROP_FRAME_COUNT)
        self.fps = self.vc.get(cv.CAP_PROP_FPS)
        self.frame_height = int(self.vc.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.vc.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_size = (self.frame_height, self.frame_width)
        #self.setGlobalMask()

    def setGlobalMask(self):
        ret, image = self.vc.read()
        mask_position = cv.selectROI('Select ROI', image)
        cv.destroyWindow('Select ROI')
        self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
        self.global_mask[mask_position[1]:mask_position[1]+mask_position[3], mask_position[0]:mask_position[0]+mask_position[2]] = 255

    def getMedianFrame(self, num_frame2use=100):
        # Get median frame
        self.frame_bucket = np.zeros((num_frame2use, self.frame_height, self.frame_width, 3), dtype=np.uint8)
        current_header = 0
        ret = False
        queue = tqdm(np.round(np.linspace(int(self.num_frame/2), self.num_frame - 1000, num_frame2use)))

        print(f'Starting background model building')
        for i, frame_number in enumerate(queue):
            self.vc.set(cv.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, image = self.vc.read()
            image = cv.bitwise_and(image, image, mask=self.global_mask)
            if not ret:
                raise(BaseException(f'Can not retrieve frame # {frame_number}'))
            self.frame_bucket[i, :, :, :] = image
        print('Background model building complete')

        self.medianFrame = np.median(self.frame_bucket,axis=0).astype(np.uint8)

        # Show median frame
        # cv.imshow('Test', self.medianFrame)
        # cv.waitKey()
        # cv.destroyWindow('Test')

    def selectColors(self):
        image = self.frame_bucket[int(self.frame_bucket.shape[0]/2), :, :, :]
        zoom_position = cv.selectROI('Select Robot', image)
        cv.destroyWindow('Select Robot')
        self.robot_color = self.getColorRange(cv.resize(image[
                                                   zoom_position[1]:zoom_position[1] + zoom_position[3],
                                                   zoom_position[0]:zoom_position[0] + zoom_position[2],
                                                   :], [zoom_position[3] * 3, zoom_position[2] * 3]))
        image = cv.absdiff(image, self.medianFrame)
        self.rat_color = self.getColorRange(image)
    def getKernel(self, size):
        size = int(size)
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, (size, size),
                                        ((int((size - 1) / 2), int((size - 1) / 2))))

    def denoiseBinaryImage(self, binaryImage):
        # opening -> delete noise : erode and dilate
        # closing -> make into big object : dilate and erode
        denoisedBinaryImage = binaryImage
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_CLOSE, self.getKernel(10))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_CLOSE, self.getKernel(10))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_CLOSE, self.getKernel(10))
        denoisedBinaryImage = cv.morphologyEx(denoisedBinaryImage, cv.MORPH_ERODE, self.getKernel(3))
        return denoisedBinaryImage

    def getColorRange(self, image):
        mask_position = cv.selectROI('Select ROI for color range extraction', image)
        cv.destroyWindow('Select ROI for color range extraction')
        targetImage = image[mask_position[1]:mask_position[1]+mask_position[3], mask_position[0]:mask_position[0]+mask_position[2], :]

        b_std = np.std(targetImage[:, :, 0])
        b_mean = np.mean(targetImage[:, :, 0])
        g_std = np.std(targetImage[:, :, 1])
        g_mean = np.mean(targetImage[:, :, 1])
        r_std = np.std(targetImage[:, :, 2])
        r_mean = np.mean(targetImage[:, :, 2])

        lower_end = np.array([b_mean - 1.5*b_std, g_mean - 1.5*g_std, r_mean - 1.5*r_std])
        higher_end = np.array([b_mean + 1.5*b_std, g_mean + 1.5*g_std, r_mean + 1.5*r_std])

        return (lower_end, higher_end)

    def __findBlob(self, image, prevPoint=None):
        """
        __findBlob: from given image, apply noise filter and find blob.
        --------------------------------------------------------------------------------
        image : 3D np.array : image to process
        --------------------------------------------------------------------------------
        return list of blobs
        --------------------------------------------------------------------------------
        """
        errorFlag = False

        """
        For the robot, find the yellow parts and mean the location
        For the rat, find the white spot, use maximum likelihood method (incl. distance), 
        """

        ########################################################
        #                Find Robot Blob                       #
        ########################################################

        # 1. Find robot by color (yellow)

        robot_mask = cv.inRange(image, self.robot_color[0], self.robot_color[1])

        robot_image = cv.bitwise_and(image, image, mask=robot_mask)
        robot_gray = cv.cvtColor(robot_image, cv.COLOR_RGB2GRAY)
        robot_binary = cv.threshold(robot_gray, 50, 255, cv.THRESH_BINARY)[1]

        robot_denoise = self.denoiseBinaryImage(robot_binary)

        # cv.imshow('Test', robot_denoise)
        # cv.waitKey()
        # cv.destroyWindow('Test')

        # 2. Find all blobs, and if the size is larger than ROBOT_SIZE_THRESHOLD
        ROBOT_SIZE_THRESHOLD = 30
        cnts = cv.findContours(robot_denoise, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        if len(cnts) > 0:
            largestContourIndex = np.argsort(np.array([cv.contourArea(cnt) for cnt in cnts]))[-1:(-1-3):-1]
            largestContours = [cnts[i] for i in largestContourIndex]

            if cv.contourArea(largestContours[0]) > 40:
                robot_center = np.round(cv.minEnclosingCircle(largestContours[0])[0]).astype(int)
                # drawImage = cv.circle(drawImage, rat_center, 30, (0, 255, 0), 3)
            else:
                robot_center = prevPoint[0]
                errorFlag = True


            # centers = []
            # for cnt in largestContours:
            #     if cv.contourArea(cnt) > ROBOT_SIZE_THRESHOLD:
            #         centers.append(np.round(cv.minEnclosingCircle(cnt)[0]))
            # if len(centers) > 0:
            #     robot_center = np.mean(centers, axis=0).astype(int)
            # else:
            #     robot_center = prevPoint[0]
            #     errorFlag = True
        else:
            robot_center = prevPoint[0]
            errorFlag = True

        #drawImage = cv.circle(drawImage, robot_center, 30, (0, 0, 255), 3)

        ########################################################
        #                Find Rat Blob                         #
        ########################################################

        # 1. Find rat by color (white)
        # Make Diff
        image = cv.absdiff(image, self.medianFrame)
        rat_mask = cv.inRange(image, self.rat_color[0], self.rat_color[1])

        rat_image = cv.bitwise_and(image, image, mask=rat_mask)
        rat_gray = cv.cvtColor(rat_image, cv.COLOR_RGB2GRAY)
        rat_binary = cv.threshold(rat_gray, 40, 255, cv.THRESH_BINARY)[1]

        rat_denoise = self.denoiseBinaryImage(rat_binary)

        # 2. Find the largest contour
        cnts = cv.findContours(rat_denoise, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        if len(cnts) > 0:
            largestContourIndex = np.argsort(np.array([cv.contourArea(cnt) for cnt in cnts]))[-1:(-1 - 3):-1]
            largestContours = [cnts[i] for i in largestContourIndex]

            centers = np.array([np.round(cv.minEnclosingCircle(cnt)[0]).astype(int) for cnt in largestContours])
            area = np.array([cv.contourArea(cnt) for cnt in largestContours])
            perimeter = np.array([cv.arcLength(cnt, closed=True) for cnt in largestContours])

            animalSize = area
            animalConvexity = area / np.array([cv.contourArea(cv.convexHull(cnt)) for cnt in largestContours])
            animalCircularity = 4 * np.pi * area / (perimeter ** 2)

            foregroundModel = {
                'animalSize': {'median': 2048.25, 'sd': 445.6631035603015},
                'animalConvexity': {'median': 0.8701496520482979, 'sd': 0.15930511168478506},
                'animalCircularity': {'median': 0.4652859512418821, 'sd': 0.12903703356433005}
            }

            L_Size = np.max([
                norm.cdf(animalSize + foregroundModel['animalSize']['sd'] * 0.1,
                         foregroundModel['animalSize']['median'], foregroundModel['animalSize']['sd'])
                - norm.cdf(animalSize - foregroundModel['animalSize']['sd'] * 0.1,
                           foregroundModel['animalSize']['median'], foregroundModel['animalSize']['sd']),
                1e-10 * np.ones(animalSize.shape)], axis=0)
            L_Convexity = np.max([
                norm.cdf(animalConvexity + foregroundModel['animalConvexity']['sd'] * 0.1,
                         foregroundModel['animalConvexity']['median'],
                         foregroundModel['animalConvexity']['sd'])
                - norm.cdf(animalConvexity - foregroundModel['animalConvexity']['sd'] * 0.1,
                           foregroundModel['animalConvexity']['median'],
                           foregroundModel['animalConvexity']['sd']),
                1e-10 * np.ones(animalConvexity.shape)], axis=0)
            L_Circularity = np.max([
                norm.cdf(animalCircularity + foregroundModel['animalCircularity']['sd'] * 0.1,
                         foregroundModel['animalCircularity']['median'],
                         foregroundModel['animalCircularity']['sd'])
                - norm.cdf(animalCircularity - foregroundModel['animalCircularity']['sd'] * 0.1,
                           foregroundModel['animalCircularity']['median'],
                           foregroundModel['animalCircularity']['sd']),
                1e-10 * np.ones(animalCircularity.shape)], axis=0)

            likelihoods = np.log(L_Size) + np.log(L_Convexity) + np.log(L_Circularity)

            rat_center = centers[np.argmax(likelihoods)]

        else:
            rat_center = prevPoint[1]
            errorFlag = True

        if errorFlag:
            self.cumerror += 1

        return (robot_center, rat_center)

    def save(self):
        np.savetxt((self.video_path.parent / 'tracking.csv').absolute(), self.output_data, '%d', delimiter=',')

    def run(self, stride):
        self.output_data = np.zeros((int(np.ceil(self.num_frame / stride)), 5), dtype=int)
        self.cumerror = 0

        # set for multiprocessing. reading frame automatically starts from this function
        self.startROIextractionThread(stride=stride)

        for idx in tqdm(range(int(np.ceil(self.num_frame / stride)))):
            try:
                frame_number, blob_centers = self.blobQ.get()
            except queue.Empty:
                print(f'Can not get frame from index {idx} of {int(np.ceil(self.num_frame / stride)) - 1}')
                self.output_data = self.output_data[:idx, :]
                break

            self.output_data[idx, :] = np.hstack((np.array(frame_number), blob_centers[0], blob_centers[1]))
        print(f'Done')
        self.isProcessed = True

    def startROIextractionThread(self, stride=5):
        """
        startROIextractionThread : start ROI extraction Thread for continuous processing.
            When called, two thread (video read and opencv ROI detection) is initiated.
            Processed ROI is stored in self.blobQ
        --------------------------------------------------------------------------------
        stride : integer : The function read one frame from from every (stride) number of frames
        """
        # Video IO Thread and Queue
        self.frameQ = Queue(maxsize=200)
        self.vcIOthread = Thread(target=self.__readVideo, args=(stride,))
        self.vcIOthread.daemon = True # indicate helper thread
        if not self.vcIOthread.is_alive():
            self.vcIOthread.start()

        # ROI detection Thread and Queue
        self.blobQ = Queue(maxsize=200)
        self.prevPoint = ((int(self.frame_height/2), int(self.frame_width/2)),(int(self.frame_height/2), int(self.frame_width/2)))
        self.roiDetectionThread = Thread(target=self.__processFrames, args=())
        self.roiDetectionThread.start()

        self.isMultithreading = True

    def __readVideo(self, stride):
        """
        __readVideo : multithreading. read video, extract frame and store in self.frameQ
        """
        print('ROI_image_stream : Video IO Thread started\n')
        self._rewindPlayHeader()
        while True:
            if not self.frameQ.full():
                if self.cur_header >= self.num_frame:
                    break
                ret, frame = self.vc.read()
                if ret == False:
                    # This is the end of the frame. num_frame is wrong!
                    print(f'Wrong number of numframe. {self.cur_header} frame does not exist!')
                    break
                frame = cv.bitwise_and(frame, frame, mask=self.global_mask)
                self.frameQ.put((self.cur_header, frame))
                self.cur_header += 1
                # skip other frames
                for i in range(stride-1):
                    ret = self.vc.grab()
                    self.cur_header += 1
            else:
                time.sleep(0.1)
        self.readVideoComplete = True
        print('ROI_image_stream : Video IO Thread stopped\n')
        return

    def __processFrames(self):
        """
        __processFrames : multithreading. extract ROI from frame stored in self.frameQ and store in self.blobQ
        """
        print('ROI_image_stream : ROI extraction Thread started\n')
        # run until frameQ is empty and thread is dead 
        while not(self.frameQ.empty()) or self.vcIOthread.is_alive():
            if not self.blobQ.full():
                frame_number, image = self.frameQ.get()
                if frame_number == 45240:
                    print('a')
                detected_blob_centers = self.__findBlob(image, prevPoint=self.prevPoint)
                self.blobQ.put((frame_number, detected_blob_centers))
                if detected_blob_centers is not None:
                    self.prevPoint = detected_blob_centers 
            else:
                time.sleep(0.1)
        print('ROI_image_stream : ROI extraction Thread stopped\n')
        return

    def _rewindPlayHeader(self):
        """
        _rewindPlayHeader : rewind the play header of the VideoCapture object
        """
        self.vc.set(cv.CAP_PROP_POS_FRAMES, 0)
        if self.vc.get(cv.CAP_PROP_POS_FRAMES) != 0:
            raise(BaseException('ROI_image_stream : Can not set the play header to the beginning'))
        self.cur_header = 0



if __name__ == "__main__":
    rrr = RobotRatRolling()
    rrr.global_mask = np.zeros(rrr.frame_size, dtype=np.uint8)
    rrr.global_mask[150:150 + 800, 335:335 + 1185] = 255
    rrr.getMedianFrame()
    rrr.selectColors()
    rrr.run(12)
    rrr.save()
    kn = teleknock.teleknock()
    kn.sendMsg("DONE")

