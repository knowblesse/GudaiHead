%% TrackingDataAnalysis
addpath('..');

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\';

%% Run
session = 1;

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

%% betweenDistanceHead
figure();
clf;
betweenDistance_noHead1 = getBetweenDistance(trackingData(R_noHead1Index, :));
betweenDistance_yesHead = getBetweenDistance(trackingData(R_yesHeadIndex, :));
betweenDistance_noHead2 = getBetweenDistance(trackingData(R_noHead2Index, :));

subplot(1,3,1);
histogram(betweenDistance_noHead1);
subplot(1,3,2);
histogram(betweenDistance_yesHead);
subplot(1,3,3);
histogram(betweenDistance_noHead2);

%% between distance with velocity
figure(2);
clf;
plot(duration(seconds(trackingData(R_noHead1Index, 1)/60), 'Format', 'mm:ss'),betweenDistance_noHead1);
    
function output = getBetweenDistance(xyxyData)
    output = ( (xyxyData(:, 2) - xyxyData(:, 3)).^2 + (xyxyData(:, 4) - xyxyData(:, 5)).^2 ).^0.5;
end







