import queue
from queue import Queue
from threading import Thread
import cv2 as cv
import numpy as np
from pathlib import Path
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
import time
import teleknock
from scipy.stats import norm


class RobotRatRolling():
    def __init__(self, path=[]):
        # Get Video Path
        if path:
            self.video_path = path
        else:
            self.video_path = Path(askopenfilename())
        self.vc = cv.VideoCapture(str(self.video_path.absolute()))
        print(f"Video Path : {self.video_path}")

        # Get Video infor
        self.num_frame = self.vc.get(cv.CAP_PROP_FRAME_COUNT)
        self.fps = self.vc.get(cv.CAP_PROP_FPS)
        self.frame_height = int(self.vc.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.vc.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_size = (self.frame_height, self.frame_width)
        #self.setGlobalMask()

        self.isPaintROIOpen = False

    def paintROI(self, image, initialState=False):
        """
        left paint : Select
        right paint : Deselect
        mouse wheel : change brush size
        """
        image_orig = image.copy()
        height = image.shape[0]
        width = image.shape[1]
        mask = np.logical_or(np.zeros((height, width), dtype=bool), initialState)
        state = {'button': 0, 'brushSize': 4, 'currentPosition': [0, 0]}  # 0: no button, 1: left, 2: right

        def mouseCallBack(event, x, y, f, state):
            if event == cv.EVENT_LBUTTONDOWN:
                state['button'] = 1
            elif event == cv.EVENT_MOUSEMOVE:
                state['currentPosition'] = [x, y]  # for brush size box
                if state['button'] == 1:
                    mask[
                    max(0, y - state['brushSize']):min(height, max(0, y + state['brushSize'])),
                    max(0, x - state['brushSize']):min(width, max(0, x + state['brushSize']))] = True
                elif state['button'] == 2:
                    mask[
                    max(0, y - state['brushSize']):min(height, y + state['brushSize']),
                    max(0, x - state['brushSize']):min(width, x + state['brushSize'])] = False

            elif event == cv.EVENT_LBUTTONUP:
                state['button'] = 0
            elif event == cv.EVENT_RBUTTONDOWN:
                state['button'] = 2
            elif event == cv.EVENT_RBUTTONUP:
                state['button'] = 0
            elif event == cv.EVENT_MOUSEWHEEL:
                if f < 0:
                    state['brushSize'] = state['brushSize'] + 1
                else:
                    state['brushSize'] = max(0, state['brushSize'] - 1)

        cv.namedWindow('Paint ROI')
        self.isPaintROIOpen = True
        cv.setMouseCallback('Paint ROI', mouseCallBack, state)
        key = -1
        while key == -1:
            image = image_orig.copy()
            image[mask, 0] = np.round(image[mask, 0] * 0.9)
            image[mask, 1] = np.round(image[mask, 1] * 0.9)
            image[mask, 2] = 255
            cv.rectangle(image,
                         (state['currentPosition'][0] - state['brushSize'],
                          state['currentPosition'][1] - state['brushSize']),
                         (state['currentPosition'][0] + state['brushSize'],
                          state['currentPosition'][1] + state['brushSize']),
                         thickness=1,
                         color=(0, 0, 0))
            cv.imshow('Paint ROI', image)
            key = cv.waitKey(1)
        cv.destroyWindow('Paint ROI')
        self.isPaintROIOpen = False
        return mask

    def concatImages(self, images, row=3, col=4):
        MAX_LENGTH = 1000
        # Check input data
        if len(images) != row * col:
            raise IndexError("Image number and row, col values do not match!")
        for image in images:
            if images[0].shape != image.shape:
                raise IndexError("Image number and row, col values do not match!")

        # Set resize factor
        height = images[0].shape[0]
        width = images[0].shape[1]

        outputHeight = height*row
        outputWidth = width*col

        zoomRatio = min(MAX_LENGTH / outputHeight, MAX_LENGTH / outputWidth)

        height_resized = np.round(height * zoomRatio).astype(int)
        width_resized = np.round(width * zoomRatio).astype(int)

        outputHeight = height_resized*row
        outputWidth = width_resized*col

        outputImage = np.zeros((outputHeight, outputWidth,3), dtype=np.uint8)
        iterImage = iter(images)
        for c in range(col):
            for r in range(row):
                outputImage[height_resized*r:height_resized*(r+1), width_resized*c:width_resized*(c+1),:] = cv.resize(next(iterImage), (width_resized, height_resized))
        return outputImage, (height_resized, width_resized)

    def setGlobalMask(self):
        ret, image = self.vc.read()
        mask_position = cv.selectROI('Select ROI', image)
        cv.destroyWindow('Select ROI')
        self.global_mask = np.zeros(self.frame_size, dtype=np.uint8)
        self.global_mask[mask_position[1]:mask_position[1]+mask_position[3], mask_position[0]:mask_position[0]+mask_position[2]] = 255

    def getMedianFrame(self, num_frame2use=20):
        # Get frames
        self.frame_bucket = np.zeros((num_frame2use, self.frame_height, self.frame_width, 3), dtype=np.uint8)
        queue = tqdm(np.round(np.linspace(0, self.num_frame-10, num_frame2use)))

        for i, frame_number in enumerate(queue):
            self.vc.set(cv.CAP_PROP_POS_FRAMES, int(frame_number))
            ret, image = self.vc.read()
            image = cv.bitwise_and(image, image, mask=self.global_mask)
            if not ret:
                raise(BaseException(f'Can not retrieve frame # {frame_number}'))
            self.frame_bucket[i, :, :, :] = image


        # Concat frame bucket into one big image
        frameSize = list(image.shape)
        frameList = [self.frame_bucket[i, :, :, :] for i in range(num_frame2use)]
        row = 4
        col = 5
        concatFrame, (height_resized, width_resized) = self.concatImages(frameList, row=row, col=col)

        # Generate Background
        """
        - Logic
        1. Show multiple frames in a 'one big image' by using `self.concatImages` function. <Select Frame>
        2. Extra window shows the current background image. <Background>
        3. If a user click a frame in the <Select Image> window, then another window is opened. <Select Background>
        4. The user select part of the image where moving objects are located (ex. robot, rat).
        5. When press space in <Select Background>, the frame without the selected part is registered as the background.
        6.  
        """

        def mouseCallBack_select_frame(event, x, y, f, vars):
            if event == cv.EVENT_LBUTTONUP:
                if self.isPaintROIOpen:
                    cv.destroyWindow('Paint ROI')
                    return
                selectedIndex = np.floor(y / height_resized).astype(int) + row * np.floor(x / width_resized).astype(int)
                # TODO : Check if the mouse up point is out of range
                mask = self.paintROI(self.frame_bucket[selectedIndex, :, :, :], initialState=False)

                frame = self.frame_bucket[selectedIndex, :, :, :].copy().astype(float)
                frame[mask,:] = np.nan
                frame = np.expand_dims(frame, axis=0)
                self.background = np.concatenate([self.background, frame], axis=0)
                if self.background.shape[0] >=2:
                    if np.any(np.isnan(np.nanmedian(self.background, axis=0))): # if any pixel is all nan in all selected frame,
                        redArray = np.zeros(frame.shape, dtype=np.uint8)
                        redArray[0, np.all(np.isnan(self.background), axis=0)[:,:,0], 2] = 255
                        cv.imshow('Background', np.round(np.nanmedian(np.concatenate([self.background, redArray], axis=0), axis=0)).astype(np.uint8))
                    else:
                        cv.imshow('Background', np.round(np.nanmedian(self.background, axis=0)).astype(np.uint8))


        self.background = np.empty([0]+frameSize)

        cv.namedWindow('Select Frame')
        cv.setMouseCallback('Select Frame', mouseCallBack_select_frame)
        cv.imshow('Select Frame', concatFrame.astype(np.uint8))

        cv.namedWindow('Background')
        cv.imshow('Background', np.zeros(image.shape))
        key = -1
        while key == -1:
            key = cv.waitKey()
        self.background = np.round(np.nanmedian(self.background, axis=0)).astype(np.uint8)
        cv.destroyWindow('Select Frame')
        cv.destroyWindow('Background')

    def selectColors(self):
        image = self.frame_bucket[int(self.frame_bucket.shape[0]/2), :, :, :]

        # For Robot, zoom!
        zoom_position = cv.selectROI('Select Robot', image)
        cv.destroyWindow('Select Robot')

        self.robot_color  = self.getColorRange(cv.resize(image[
            zoom_position[1]:zoom_position[1] + zoom_position[3],
            zoom_position[0]:zoom_position[0] + zoom_position[2],
            :], [zoom_position[3] * 3, zoom_position[2] * 3]), std=1.5)

        # For Rat
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

    def getColorRange(self, image, std=1):
        mask = self.paintROI(image)
        lower_end = np.max([[0, 0, 0], np.round(np.mean(image[mask,:],axis=0) - std*np.std(image[mask, :], axis=0))], axis=0).astype(np.uint8)
        higher_end = np.min([[255, 255, 255], np.round(np.mean(image[mask,:],axis=0) + std*np.std(image[mask, :], axis=0))], axis=0).astype(np.uint8)
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

        foregroundMask = \
            cv.threshold(cv.cvtColor(cv.absdiff(image, self.background), cv.COLOR_RGB2GRAY), 30, 255, cv.THRESH_BINARY)[1]

        image = cv.bitwise_and(image, image, mask=foregroundMask)

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

            # Get Median value
            centers = []
            for cnt in largestContours:
                if cv.contourArea(cnt) > ROBOT_SIZE_THRESHOLD:
                    centers.append(np.round(cv.minEnclosingCircle(cnt)[0]))
            if len(centers) > 0:
                robot_center = np.median(centers, axis=0).astype(int)
            else:
                robot_center = prevPoint[0]
                errorFlag = True
        else:
            robot_center = prevPoint[0]
            errorFlag = True

        #drawImage = cv.circle(drawImage, robot_center, 30, (0, 0, 255), 3)

        ########################################################
        #                Find Rat Blob                         #
        ########################################################

        # 1. Find rat by color (white)
        rat_mask = cv.inRange(image, self.rat_color[0], self.rat_color[1])

        rat_image = cv.bitwise_and(image, image, mask=rat_mask)
        rat_gray = cv.cvtColor(rat_image, cv.COLOR_RGB2GRAY)
        rat_binary = cv.threshold(rat_gray, 40, 255, cv.THRESH_BINARY)[1]

        rat_denoise = self.denoiseBinaryImage(rat_binary)

        # 2. Find the largest contour
        cnts = cv.findContours(rat_denoise, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        if len(cnts) > 0:
            largestContourIndex = np.argsort(np.array([cv.contourArea(cnt) for cnt in cnts]))[-1:(-1 - 3):-1]
            sizeSortedContours = [cnts[i] for i in largestContourIndex]
            rat_center = np.round(cv.minEnclosingCircle(sizeSortedContours[0])[0]).astype(int)

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

