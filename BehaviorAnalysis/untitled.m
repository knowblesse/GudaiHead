%% TrackingDataAnalysis
addpath('..');

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest2\';

%% Batch
OUT = table(zeros(10,2), zeros(10,2), zeros(10,2), 'VariableNames', ["AverageVelocity", "CenterPercent", "BetweenDistance"]);
for session = 1 : 10
    
    folderPath = fullfile(BASEPATH, strcat('R', num2str(session, '%02d')));

    %% Read tracking.csv
    trackingData = readmatrix(glob(folderPath, '.*.csv', true));
    timestamp = trackingData(:,1) / FPS; % assume constant frame rate

    %% Read time.txt
    timeData = readlines(glob(folderPath, '.*.txt', true));
    separator = seconds(duration(timeData, 'InputFormat', 'mm:ss'));
    endOfLastNoHeadTime = timestamp(end); % else, use the last tracking time.
    separator = [separator; endOfLastNoHeadTime];
    timestampIndex = [1; arrayfun(@(x) find(timestamp>=x, 1), separator)];

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

end


%% In script functions
function output = isCenter(xyData)
    % Define center: 
    x = xyData(:,1);
    y = xyData(:,2);
    
    output = (552 < x & x < 1380) & (360 < y & y < 895);
end
