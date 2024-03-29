%% TrackingDataAnalysis_Train
addpath('..');
load("apparatus.mat");

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest1';
SUBJADD = 0; % for R11-20, use value of 10. 
px2cm = 0.2171;

%% Batch
out_velocity = zeros(10, 10);
out_center_time = zeros(10, 10);
out_corner_time = zeros(10, 10);
out_mean_btw_distance = zeros(10, 10);
out_median_btw_distance = zeros(10, 10);
out_freezing = zeros(10,10);
out_min_run = zeros(10,10);
out_view_ratio = zeros(10, 10);


for session = 1 : 10
    
    folderPath = fullfile(BASEPATH, strcat('R', num2str(session + SUBJADD,'%02d')));
    
    fprintf('\n%d\n', session);
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

    % Additional Front/End data
    timeRange.def1A = [separator_raw(1), round(separator_raw(1) + (separator_raw(2) - separator_raw(1))/2)];
    timeRange.def1B = [round(separator_raw(1) + (separator_raw(2) - separator_raw(1))/2), separator_raw(2)];
    timeRange.infA = [separator_raw(3), round(separator_raw(3) + (separator_raw(4) - separator_raw(3))/2)];
    timeRange.infB = [round(separator_raw(3) + (separator_raw(4) - separator_raw(3))/2), separator_raw(4)];

    %% Positions 
    % [Caution] concatenated video only affect velocity.
    %   So this part must preceed the exception handling
    ratPosition_all = [...
        movmean(butterData(:,3),5),...
        movmean(butterData(:,2),5)]; % x, y
    robotPosition_all = [...
        movmean(trackingData(:,2),10),...
        movmean(trackingData(:,3),10)]; % x, y
    
    %% Degree
    prev_head_direction = butterData(1,4);
    degree_offset_value = zeros(size(butterData, 1),1);
    for bd = 2 : size(butterData,1)
        if abs(butterData(bd,4) - prev_head_direction) > 180
            if butterData(bd,4) > prev_head_direction
                degree_offset_value(bd:end) = degree_offset_value(bd:end) - 360;
            else
                degree_offset_value(bd:end) = degree_offset_value(bd:end) + 360;
            end
        end
        prev_head_direction = butterData(bd,4);
    end

    ratHeadDegree_all = movmean(butterData(:, 4) + degree_offset_value, 5);
    
    %% Exception Handling
    if contains(folderPath, 'Rtest1') && contains(folderPath, 'R03')
        % Test1 - R3 has two Inflated Head phase
        timeRange.inf = [separator_raw(5), separator_raw(6)];
        timeRange.def2 = [separator_raw(7), separator_raw(8)];
    end
    velocityIntegrity_all = false(size(ratPosition_all,1),1); % if true, don't use the distance
    if contains(folderPath, 'Rtest3') && (...
            contains(folderPath, 'R11') ||...
            contains(folderPath, 'R16') ||...
            contains(folderPath, 'R18') ||...
            contains(folderPath, 'R20') )
        % Test3 - R11, 16,  : concatenated two video
        [~, clippedPart] = max(diff(butterData(:,2)));
        ratPosition_all = [...
            [...
                movmean(butterData(1:clippedPart,3),5);...
                movmean(butterData(clippedPart+1:end,3),5)...
            ],...
            [...
                movmean(butterData(1:clippedPart,2),5);...
                movmean(butterData(clippedPart+1:end,2),5)
            ]];

        ratHeadDegree_all = [...
                movmean(butterData(:, 4) + degree_offset_value,5);...
                movmean(butterData(:, 4) + degree_offset_value,5)...
            ];

        robotPosition_all = [...
            [...
                movmean(trackingData(1:clippedPart,3),10);...
                movmean(trackingData(clippedPart+1:end,3),10)...
            ],...
            [...
                movmean(trackingData(1:clippedPart,2),10);...
                movmean(trackingData(clippedPart+1:end,2),10)
            ]];
        velocityIntegrity_all(clippedPart) = true;
    end

    ratHeadDegree_all = rem(ratHeadDegree_all + 36000, 360);

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

    % Additional Front/End data
    totalTime.def1A = diff(timeRange.def1A);
    totalTime.def1B = diff(timeRange.def1B);
    totalTime.infA = diff(timeRange.infA);
    totalTime.infB = diff(timeRange.infB);

    %% Calculate results
    eventList = ["hab", "def1", "p1", "inf", "p2", "def2", "def1A", "def1B", "infA", "infB"];

    for i = 1 : 10
        event = eventList(i);
        ratPosition = ratPosition_all(dataIndex.(event)(1): dataIndex.(event)(2), :); % x, y
        ratHeadDegree = ratHeadDegree_all(dataIndex.(event)(1): dataIndex.(event)(2), :); % x, y
        robotPosition = robotPosition_all(dataIndex.(event)(1): dataIndex.(event)(2), :); % x, y
        timestamp_event = timestamp(dataIndex.(event)(1): dataIndex.(event)(2));
        velocityIntegrity = velocityIntegrity_all(dataIndex.(event)(1): dataIndex.(event)(2));
        
        % Velocity
        deltaPosition = sum(diff(ratPosition).^2,2).^0.5;
        if (any(velocityIntegrity))
            idx = find(velocityIntegrity);
            deltaPosition(idx) = deltaPosition(idx-1);
        end
        out_velocity(session, i) = sum(deltaPosition) / totalTime.(event) * px2cm;
        
        % Center Ratio
        output_ = false(size(ratPosition,1),1);
        for pIdx = 1 : size(ratPosition,1)
            output_(pIdx) = apparatus.center(round(ratPosition(pIdx,2)), round(ratPosition(pIdx,1)),bgNumber);
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

        if false % [DEBUG] draw freezing graph
            figure();
            plot(freezing_timestamp, isFreezing*5, 'LineWidth',1);
            hold on;
            plot(freezing_timestamp, meanVelocity, 'LineWidth',1);
        end

        %% Minimum range to escape
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
        minVals = [];
        for val = vals
            minVals = [minVals, min(btwDistance(max(val-3-LPS, 1) : val-3, 1))];
        end
        out_min_run(session, i) = mean(minVals);

        if false % [DEBUG] Draw sudden run aways
            figure();
            plot(deltaPosition);
            hold on;
            plot(btwDistance)
            v = scatter(vals, deltaPosition(vals), 'filled', 'r');
        end

        %% isViewing
        % If the rat is too close to the robot (for example 30cm), 
        % the head direction tend to misclassified due to the robot image.
        % Therefore, isView value is only calculated when the distance
        % between the robot and the rat is at least 30cm. 
        ratViewRange = ratHeadDegree + [-30, + 30];
        fromRatToRobot = robotPosition - ratPosition;
        isViewing = false(size(ratPosition,1),1);
        isViewTotalCount = 0;
        for pIdx = 1 : size(ratPosition,1)
            if false % [DEBUG] show direction plot
                figure();
                imshow(apparatus.image);
                hold on;
                scatter(ratPosition(pIdx,1), ratPosition(pIdx,2), 'filled', 'g');
                plot(...
                    [ratPosition(pIdx,1), ratPosition(pIdx,1) + 30*cosd(ratHeadDegree(pIdx))],...
                    [ratPosition(pIdx,2), ratPosition(pIdx,2) + 30*sind(ratHeadDegree(pIdx))], 'r')
                scatter(robotPosition(pIdx,1), robotPosition(pIdx,2), 'filled', 'r');
                plot(...
                    [ratPosition(pIdx,1), ratPosition(pIdx,1) + fromRatToRobot(pIdx,1)],...
                    [ratPosition(pIdx,2), ratPosition(pIdx,2) + fromRatToRobot(pIdx,2)], '--');
                title(string(duration(seconds(timestamp_event(pIdx)), 'Format', 'mm:ss')));
            end

            if btwDistance(pIdx) >= 20
                isViewTotalCount = isViewTotalCount + 1;
                relativeRobotAngle = atan2d(fromRatToRobot(pIdx, 2), fromRatToRobot(pIdx, 1));
                if relativeRobotAngle < 0
                    relativeRobotAngle = relativeRobotAngle + 360;
                end
            
                if relativeRobotAngle > ratViewRange(pIdx, 2)
                    if relativeRobotAngle > ratViewRange(pIdx, 1) + 360
                        inFOV = true;
                    else
                        inFOV = false;
                    end
                else
                    if relativeRobotAngle > ratViewRange(pIdx, 1)
                        inFOV = true;
                    else
                        inFOV = false;
                    end
                end

                isViewing(pIdx) = inFOV;
            end

        end

        out_view_ratio(session, i) = sum(isViewing) / isViewTotalCount;
        fprintf('%s : %d / %d\n', eventList(i), sum(isViewing), isViewTotalCount);

    end

end

out_center_time = out_center_time * 100;
out_corner_time = out_corner_time * 100;
out_freezing = out_freezing * 100;
out_view_ratio = out_view_ratio * 100;

