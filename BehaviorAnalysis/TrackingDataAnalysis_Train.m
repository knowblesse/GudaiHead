%% TrackingDataAnalysis_Train
addpath('..');

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest1';
px2cm = 0.2171;

%% Batch

out_velocity = zeros(10, 6);
out_center_time = zeros(10, 6);
out_mean_btw_distance = zeros(10, 6);
out_median_btw_distance = zeros(10, 6);

for session = 1 : 10

    folderPath = fullfile(BASEPATH, strcat('R', num2str(session,'%02d')));

    %% Read tracking.csv for robot location and buttered.csv for fine rat location
    trackingData = readmatrix(glob(folderPath, 'tracking.csv', true));
    butterData = readmatrix(glob(folderPath, '.*buttered.csv', true), 'Delimiter','\t');

    if size(trackingData,1) ~= size(butterData,1)
        error('tracking and butter data size mismatch');
    end

    %% Calculate timestamp
    timestamp = trackingData(:,1) / FPS; % assume constant frame rate

    %% Read time.txt
    % 1: End of Hab
    % 2: End of def1
    % 3: End of pause1
    % 4: End of inf
    % 5: End of def2
    timeData = readlines(glob(folderPath, 'time.txt', true));
    timeData = timeData(~arrayfun(@(X) X=="", timeData)); % remove empty lines
    separator_raw = seconds(duration(timeData, 'InputFormat', 'mm:ss'));
    
    %% Set Time Range (tr)
    timeRange.hab = [0, separator_raw(1)];
    timeRange.def1 = [separator_raw(1), separator_raw(2)];
    timeRange.p1 = [separator_raw(2), separator_raw(3)];
    timeRange.inf = [separator_raw(3), separator_raw(4)];
    timeRange.p2 = [separator_raw(4), separator_raw(5)];
    timeRange.def2 = [separator_raw(5), timestamp(end)];
    
    %% Exception Handling
    if contains(BASEPATH, 'Rtest1') && contains(BASEPATH, 'R03')
        % Test1 - R3 has two Inflated Head phase
        timeRange.def_robot2 = [separator_raw(7), separator_raw(8)];
    end

    %% Get Data Indices
    names = string(fieldnames(timeRange))';
    for name = names
        dataIndex.(name) = arrayfun(@(x) find(timestamp>=x, 1), timeRange.(name));
    end

    %% Calculate Times
    totalTime.hab = diff(timeRange.hab);
    totalTime.def1 = diff(timeRange.def1);
    totalTime.p1 = diff(timeRange.p1);
    totalTime.inf = diff(timeRange.inf);
    totalTime.p2 = diff(timeRange.p2);
    totalTime.def2 = diff(timeRange.def2);
    
    %% TODO: If video is concatenated, ignore big excursion

    %% Calculate results
    eventList = ["hab", "def1", "p1", "inf", "p2", "def2"];

    for i = 1 : 6
        event = eventList(i);
        ratPosition = [...
            movmean(butterData(dataIndex.(event)(1): dataIndex.(event)(2),3),5),...
            movmean(butterData(dataIndex.(event)(1): dataIndex.(event)(2),2),5)]; % x, y
        robotPosition = [...
            movmean(trackingData(dataIndex.(event)(1): dataIndex.(event)(2),2),10),...
            movmean(trackingData(dataIndex.(event)(1): dataIndex.(event)(2),3),10)]; % x, y


        clf;
        imshow(apparatus.image);
        hold on;
        plot(ratPosition(:,1), ratPosition(:,2), 'LineWidth',1, 'Color', 'w');
        plot(robotPosition(:,1), robotPosition(:,2), 'LineWidth',1, 'Color', 'r');



        ratPosition = [...
            movmean(butterData(:,3),5),...
            movmean(butterData(:,2),5)]; % x, y
        robotPosition = [...
            movmean(trackingData(:,2),10),...
            movmean(trackingData(:,3),10)]; % x, y
    
        % Velocity
        out_velocity(session, i) = sum(sum(diff(ratPosition).^2,2).^0.5) / totalTime.(event) * px2cm;
        
        % Center time
        out_center_time(session,i) = mean(isCenter(ratPosition) .* (1/LPS));

        % Calculate between distance
        out_mean_btw_distance(session,i) = mean(sum((ratPosition - robotPosition).^2, 2) .^0.5);
        out_median_btw_distance(session,i) = median(sum((ratPosition - robotPosition).^2, 2) .^0.5);
    
        % Minimum range to escape

        % approach
    end

end


%% In script functions
function output = isCenter(xyData)
    % Define center: 
    x = xyData(:,1);
    y = xyData(:,2);
    
    output = (552 < x & x < 1380) & (360 < y & y < 895);
end
