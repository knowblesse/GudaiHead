%% TrackingDataAnalysis_Test
addpath('..');

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest2';

%% Batch
OUT = table(zeros(10,2), zeros(10,2), zeros(10,2), 'VariableNames', ["AverageVelocity", "CenterPercent", "BetweenDistance"]);
FOV = zeros(10,2);
for session = 1 : 10
    
    folderPath = fullfile(BASEPATH, strcat('R', num2str(session, '%02d')));

    %% Read tracking.csv
    trackingData = readmatrix(glob(folderPath, 'tracking.csv', true));
    timestamp = trackingData(:,1) / FPS; % assume constant frame rate

    %% Read time.txt
    timeData = readlines(glob(folderPath, '.*.txt', true));
    separator = seconds(duration(timeData, 'InputFormat', 'mm:ss'));
    endOfLastNoHeadTime = timestamp(end); % else, use the last tracking time.
    separator = [separator; endOfLastNoHeadTime];
    timestampIndex = [1; arrayfun(@(x) find(timestamp>=x, 1), separator)];

    %% Read Head direction
    headDegreePath = glob(folderPath, '.*buttered.csv', true);
    if isempty(headDegreePath)
        isHeadDegreePresent = false;
    else
        isHeadDegreePresent = true;
        headDegree = readmatrix(headDegreePath); % frameNumber, row, col, degree
    end

    %% Process index
    H_index = timestampIndex(1):timestampIndex(2);
    R_index = timestampIndex(2):timestampIndex(3);

    %% Process Time
    totalTime_H = separator(1);
    totalTime_R = endOfLastNoHeadTime - separator(1);
    
    %% Calculate Center time
    centerTime_H = sum(isCenter(trackingData(H_index, 4:5))) * (1/LPS);
    centerTime_R = sum(isCenter(trackingData(R_index, 4:5))) * (1/LPS);

    OUT.CenterPercent(session,:) = [...
        centerTime_H / totalTime_H,...
        centerTime_R / totalTime_R...
        ] * 100;

    %% Calculate mean velocity
    distance_H = sum(sum(diff(trackingData(H_index, 4:5)).^2,2).^0.5);
    distance_R = sum(sum(diff(trackingData(R_index, 4:5)).^2,2).^0.5);

    OUT.AverageVelocity(session,:) = [...
        distance_H / totalTime_H,...
        distance_R / totalTime_R...
        ];

    %% Calculate mean between distance
    betweenDistance_H = mean(sum((trackingData(H_index, 2:3) - trackingData(H_index, 4:5)).^2, 2) .^0.5);
    betweenDistance_R = mean(sum((trackingData(R_index, 2:3) - trackingData(R_index, 4:5)).^2, 2) .^0.5);

    OUT.BetweenDistance(session,:) = [betweenDistance_H, betweenDistance_R];

    %% Calculate is Viewing
    if isHeadDegreePresent
    inFOV_H = zeros(numel(H_index),1);
    inFOV_R = zeros(numel(R_index),1);
        for i = 1 : numel(H_index)
            idx = H_index(i);
            
            ratHeadDegree = headDegree(idx,4);
            ratViewRange = ratHeadDegree + [-30, + 30];

            fromRatToRobot = [...
                trackingData(idx, 2) - trackingData(idx, 4),...
                trackingData(idx, 3) - trackingData(idx, 5)...
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
            inFOV_H(i) = inFOV;
        end

        for i = 1 : numel(R_index)
            idx = R_index(i);
            
            ratHeadDegree = headDegree(idx,4);
            ratViewRange = ratHeadDegree + [-30, + 30];

            fromRatToRobot = [...
                trackingData(idx, 2) - trackingData(idx, 4),...
                trackingData(idx, 3) - trackingData(idx, 5)...
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
            inFOV_R(i) = inFOV;
        end
    FOV(session, :) = [sum(inFOV_H) / totalTime_H, sum(inFOV_R) / totalTime_R];
    end
end


%% In script functions
function output = isCenter(xyData)
    % Define center: 
    x = xyData(:,1);
    y = xyData(:,2);
    
    output = (552 < x & x < 1380) & (360 < y & y < 895);
end
