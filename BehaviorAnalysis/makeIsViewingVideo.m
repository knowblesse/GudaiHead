folderPath = "C:\Users\knowb\SynologyDrive\Temp";

FPS = 60;
LPS = 5;

%% Read tracking.csv
trackingData = readmatrix(glob(folderPath, 'tracking.csv', true));
timestamp = trackingData(:,1) / FPS; % assume constant frame rate

%% Read Head direction
headDegreePath = glob(folderPath, '.*buttered.csv', true);
headDegree = readmatrix(headDegreePath); % frameNumber, row, col, degree

%% Make Video Reader
vr = VideoReader(glob(folderPath, '.*.mkv', true));
vw = VideoWriter(fullfile(folderPath, 'marked.mp4'));
vw.FrameRate = 10;
vw.open();


%% Calculate is Viewing
numFrame = headDegree(end,1);
i = 0;
figure('Visible', 'off');

inFOVData = zeros(size(timestamp));

for i = 1 : numel(timestamp)
    frame = vr.read(headDegree(i,1)+1);

    ratHeadDegree = headDegree(i,4);
    ratViewRange = ratHeadDegree + [-62, + 62];
    
    fromRatToRobot = [...
        trackingData(i, 2) - trackingData(i, 4),...
        trackingData(i, 3) - trackingData(i, 5)...
        ];
    relativeRobotAngle = atan2d(fromRatToRobot(2), fromRatToRobot(1));
    if relativeRobotAngle < 0
        relativeRobotAngle = relativeRobotAngle + 360;
    end

    if relativeRobotAngle > ratViewRange(2)
        if relativeRobotAngle > ratViewRange(1) + 360
            inFOV = true;
        else
            inFOV = false;
        end
    else
        if relativeRobotAngle > ratViewRange(1)
            inFOV = true;
        else
            inFOV = false;
        end
    end
    
    
    if inFOV
        patchColor = [0.3686    0.8627    0.1216];
    else
        patchColor = [0.5490         0    0.2039];
    end
    
    %% Draw Rat and Head angle
    clf;
    imshow(frame);
    hold on;
    scatter(headDegree(i,3), headDegree(i,2), 400, 'filled', 'r');
    
    line([headDegree(i,3), headDegree(i,3) + 100*cosd(ratHeadDegree)], [headDegree(i,2), headDegree(i,2) + 100*sind(ratHeadDegree)], 'Color', 'w', 'LineWidth',2);
    patch([...
        headDegree(i,3), headDegree(i,3) + 3000 * cosd(ratViewRange(1)), headDegree(i,3) + 3000 * cosd(ratViewRange(2))],[...
        headDegree(i,2), headDegree(i,2) + 3000 * sind(ratViewRange(1)), headDegree(i,2) + 3000 * sind(ratViewRange(2))],...
        patchColor,...
        'FaceAlpha', 0.2);
    
    inFOVData(i) = inFOV;
    
    newFrame = getframe(gca);
    vw.writeVideo(newFrame.cdata);
    fprintf('%d\n',i);
end
vw.close();
