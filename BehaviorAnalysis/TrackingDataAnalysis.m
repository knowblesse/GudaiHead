%% TrackingDataAnalysis
addpath('..');

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest1';

%% Batch
OUT = table(zeros(10,3), zeros(10,3), zeros(10,3), 'VariableNames', ["AverageVelocity", "CenterPercent", "BetweenDistance"]);
for session = 1 : 10
    
    folderPath = fullfile(BASEPATH, strcat('R', num2str(session)));

    %% Read tracking.csv
    trackingData = readmatrix(glob(folderPath, '.*.csv', true));
    timestamp = trackingData(:,1) / FPS; % assume constant frame rate

    %% Read time.txt
    % 1: End of Hab
    % 2: End of No Head Robot
    % 3: End of pause1
    % 4: End of Head Robot
    % 5: End of pause2
    timeData = readlines(glob(folderPath, '.*.txt', true));
    timeData = timeData(~arrayfun(@(X) X=="", timeData)); % remove empty lines
    separator = seconds(duration(timeData, 'InputFormat', 'mm:ss'));
    if numel(separator) > 5 % in some cases, like R3, entries contain more than 5 labels.
        % R3. 
        endOfLastNoHeadTime = separator(6); % if the label is more than 5, use the last label as the end of the experiment.
    else
        endOfLastNoHeadTime = timestamp(end); % else, use the last tracking time.
    end
    separator = [separator(1:5); endOfLastNoHeadTime];
    timestampIndex = [1; arrayfun(@(x) find(timestamp>=x, 1), separator)];

    %% Process index
    R_noHead1Index = timestampIndex(2):timestampIndex(3);
    R_yesHeadIndex = timestampIndex(4):timestampIndex(5);
    R_noHead2Index = timestampIndex(6):timestampIndex(7);

    %% Process Time
    totalTime_noHead1 = separator(2) - separator(1);
    totalTime_yesHead = separator(4) - separator(3);
    totalTime_noHead2 = separator(6) - separator(5);
    
    %% Calculate Center time
    centerTime_noHead1 = sum(isCenter(trackingData(R_noHead1Index, 4:5))) * (1/LPS);
    centerTime_yesHead = sum(isCenter(trackingData(R_yesHeadIndex, 4:5))) * (1/LPS);
    centerTime_noHead2 = sum(isCenter(trackingData(R_noHead2Index, 4:5))) * (1/LPS);

    OUT.CenterPercent(session,:) = [...
        centerTime_noHead1 / totalTime_noHead1,...
        centerTime_yesHead / totalTime_yesHead,...
        centerTime_noHead2 / totalTime_noHead2...
        ] * 100;

    %% Calculate mean velocity
    distance_noHead1 = sum(sum(diff(trackingData(R_noHead1Index, 4:5)).^2,2).^0.5);

    distance_yesHead = sum(sum(diff(trackingData(R_yesHeadIndex, 4:5)).^2,2).^0.5);

    distance_noHead2 = sum(sum(diff(trackingData(R_noHead2Index, 4:5)).^2,2).^0.5);

    OUT.AverageVelocity(session,:) = [...
        distance_noHead1 / totalTime_noHead1,...
        distance_yesHead / totalTime_yesHead,...
        distance_noHead2 / totalTime_noHead2...
        ];

    %% Calculate mean between distance
    betweenDistance_noHead1 = mean(sum((trackingData(R_noHead1Index, 2:3) - trackingData(R_noHead1Index, 4:5)).^2, 2) .^0.5);
    betweenDistance_yesHead = mean(sum((trackingData(R_yesHeadIndex, 2:3) - trackingData(R_yesHeadIndex, 4:5)).^2, 2) .^0.5);
    betweenDistance_noHead2 = mean(sum((trackingData(R_noHead2Index, 2:3) - trackingData(R_noHead2Index, 4:5)).^2, 2) .^0.5);

    OUT.BetweenDistance(session,:) = [betweenDistance_noHead1, betweenDistance_yesHead, betweenDistance_noHead2];
end


%% In script functions
function output = isCenter(xyData)
    % Define center: 
    x = xyData(:,1);
    y = xyData(:,2);
    
    output = (552 < x & x < 1380) & (360 < y & y < 895);
end
