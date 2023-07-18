%% TrackingDataAnalysis_Train
addpath('..');
load("apparatus.mat");

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest1';
px2cm = 0.2171;

%% Batch

out_velocity = zeros(10, 6);
out_center_time = zeros(10, 6);
out_corner_time = zeros(10, 6);
out_mean_btw_distance = zeros(10, 6);
out_median_btw_distance = zeros(10, 6);
out_freezing = zeros(10,6);
out_min_run = zeros(10,6);

for session = 1 : 10

    folderPath = fullfile(BASEPATH, strcat('R', num2str(session,'%02d')));

    %% Read tracking.csv for robot location and buttered.csv for fine rat location
    trackingData = readmatrix(glob(folderPath, 'tracking.csv', true));
    butterData = readmatrix(glob(folderPath, '.*buttered.csv', true), 'Delimiter','\t');
    bgNumber = str2double(regexp(glob(folderPath, 'bg\d\d.txt', false), '\d\d', 'match')); % background number

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
        timestamp_event = timestamp(dataIndex.(event)(1): dataIndex.(event)(2));
        
%         %% all
%         ratPosition = [...
%             movmean(butterData(:,3),5),...
%             movmean(butterData(:,2),5)]; % x, y
%         robotPosition = [...
%             movmean(trackingData(:,2),10),...
%             movmean(trackingData(:,3),10)]; % x, y
%         timestamp_event = timestamp;
%         
%         %% remove p1 and p2
%         a = 1 : size(butterData,1);
%         a([dataIndex.p1(1):dataIndex.p1(2), dataIndex.p2(1):dataIndex.p2(2)]) = [];
% 
%         ratPosition = [...
%             movmean(butterData(a,3),5),...
%             movmean(butterData(a,2),5)]; % x, y
%         robotPosition = [...
%             movmean(trackingData(a,2),10),...
%             movmean(trackingData(a,3),10)]; % x, y
    
        % Velocity
        deltaPosition = sum(diff(ratPosition).^2,2).^0.5;
        out_velocity(session, i) = sum(deltaPosition) / totalTime.(event) * px2cm;
        
        % Center Ratio
        output_ = false(size(ratPosition,1),1);
        for pIdx = 1 : size(ratPosition,1)
            output_(pIdx) = apparatus.edge(round(ratPosition(pIdx,2)), round(ratPosition(pIdx,1)),bgNumber);
        end
        out_center_time(session,i) = sum(output_) / numel(output_);

        % Corner Ratio
        output_ = false(size(ratPosition,1),1);
        for pIdx = 1 : size(ratPosition,1)
            output_(pIdx) = apparatus.corner(round(ratPosition(pIdx,2)), round(ratPosition(pIdx,1)),bgNumber);
        end
        out_corner_time(session,i) = sum(output_) / numel(output_);

        % Calculate between distance
        btwDistance = sum((ratPosition - robotPosition).^2, 2) .^0.5 * px2cm;
        out_mean_btw_distance(session,i) = mean(btwDistance);
        out_median_btw_distance(session,i) = median(btwDistance);

        %% Calculate freezing
        % logic : using the moving window, if deltaPosition is less than
        % 5px (=1.0855 cm / per label(0.2sec), for at least 2 seconds,
        % animal is freezing.
        windowSeconds = 2;
        windowSize = LPS * windowSeconds + 1 - 1;
        % plus one => from 0 sec to 2 sec => 11 points not 10 points
        % minus one because we are using diff value
        isFreezing = zeros(size(ratPosition,1)-windowSize,1);
        meanVelocity = zeros(size(ratPosition,1)-windowSize,1);
        for pIdx = windowSize : size(ratPosition,1)-1
            isFreezing(pIdx-windowSize+1) = all(deltaPosition(pIdx-windowSize+1:pIdx) < 5);
            meanVelocity(pIdx-windowSize+1) = mean(deltaPosition(pIdx-windowSize+1:pIdx));
        end
        freezing_timestamp = timestamp_event(windowSize+1 :end) - windowSeconds/2;
        out_freezing(session,i) = sum(isFreezing) / numel(isFreezing);

%         plot(freezing_timestamp, isFreezing*5, 'LineWidth',1);
%         hold on;
%         plot(freezing_timestamp, meanVelocity, 'LineWidth',1);

        

        % Minimum range to escape
        %clf;
        %plot(deltaPosition)
        %hold on;
        %plot(btwDistance)
        vals = [];
        pIdx = 1;
        while pIdx < size(deltaPosition,1)
            if deltaPosition(pIdx) > 30 % if rat is abruptly moved
                deltaBtwDistance = btwDistance(min(pIdx+1, size(btwDistance,1))) - btwDistance(max(pIdx-3, 1));
                if deltaBtwDistance > 20 % if distance was increased more than 30 cm during 2 sec window
                    vals = [vals, pIdx];
                    pIdx = pIdx + LPS;
                end
            end
            pIdx = pIdx + 1;
        end
        %v = scatter(vals, deltaPosition(vals), 'filled', 'r');
        minVals = [];
        for val = vals
            minVals = [minVals, min(btwDistance(max(val-3-LPS, 1) : val-3, 1))];
        end
        out_min_run(session, i) = mean(minVals);

        % Head Degree

    end

end
