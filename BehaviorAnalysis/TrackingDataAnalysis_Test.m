%% TrackingDataAnalysis_Test
addpath('..');
load("apparatus.mat");

%% Constants
FPS = 60;
LPS = 5;
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest2';
SUBJADD = 0; % for R11-20, use value of 10. 
px2cm = 0.2171;
% Test2
G1 = [1, 3, 4, 5, 6];
G2 = [2, 7, 8, 9, 10];
% Test 4
%G1 = 1:5; 
%G2 = 6:10;
%G1 = 1:5;
%G2 = 1:5;

%% Batch
out_velocity = zeros(10, 2);
out_center_time = zeros(10, 2);
out_corner_time = zeros(10, 2);
out_mean_btw_distance = zeros(10, 2);
out_median_btw_distance = zeros(10, 2);
out_freezing = zeros(10,2);
out_min_run = zeros(10,2);
out_view_ratio = zeros(10, 2);
out_around_robot_ratio = zeros(10,2);

histo_box = cell(10,1);

for session = 1 : 10
    
    folderPath = fullfile(BASEPATH, strcat('R', num2str(session + SUBJADD,'%02d')));

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
    timeData = readlines(glob(folderPath, 'time.txt', true));
    timeData = timeData(~arrayfun(@(X) X=="", timeData)); % remove empty lines
    if numel(timeData) ~= 1
        error('more than two time points');
    end
    separator_raw = seconds(duration(timeData, 'InputFormat', 'mm:ss'));
    
    timeRange.hab = [0, separator_raw(1)];
    timeRange.bot = [separator_raw(1), timestamp(end)];

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
    velocityIntegrity_all = false(size(ratPosition_all,1),1); % if true, don't use the distance
    if (contains(folderPath, 'Rtest2') && contains(folderPath, 'R02')) ||...
        (contains(folderPath, 'Rtest4') && contains(folderPath, 'R12'))
        % Test2 - R2, Test4 - R12 : concatenated two video
        [~, clippedPart] = max(abs(diff(butterData(:,2))));
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
    elseif contains(folderPath, 'Rtest5')
        robotPosition_all = repmat(trackingData(1,[2,3]), size(robotPosition_all,1),1);
        timeRange.hab = [0, timestamp(end)];
        timeRange.bot = [0, timestamp(end)];
    end

    ratHeadDegree_all = rem(ratHeadDegree_all + 36000, 360);
    
    %% Get Data Indices
    names = string(fieldnames(timeRange))';
    for name = names
        dataIndex.(name) = arrayfun(@(x) find(timestamp>=x, 1), timeRange.(name));
    end

    %% Calculate Times
    totalTime.hab = diff(timeRange.hab);
    totalTime.bot = diff(timeRange.bot);

    %% Calculate results
    eventList = ["hab", "bot"];

    for i = 1 : 2
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

        % around robot
        out_around_robot_ratio(session,i) = sum(btwDistance < 90) / numel(btwDistance);
        if event == "hab"
            histo_box{session} = btwDistance;
        end
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
        ratViewRange = ratHeadDegree + [-30, + 30];
        fromRatToRobot = robotPosition - ratPosition;
        isViewing = false(size(ratPosition,1),1);
        isViewTotalCount = 0;
        for pIdx = 1 : size(ratPosition,1)
            if false % [DEBUG] show direction plot
                figure();
                imshow(apparatus.image(:,:,:,1));
                hold on;
                scatter(ratPosition(pIdx,1), ratPosition(pIdx,2), 'filled', 'g');
                plot(...
                    [ratPosition(pIdx,1), ratPosition(pIdx,1) + 30*cosd(ratHeadDegree(pIdx))],...
                    [ratPosition(pIdx,2), ratPosition(pIdx,2) + 30*sind(ratHeadDegree(pIdx))], 'r', "LineWidth",2)
                scatter(robotPosition(pIdx,1), robotPosition(pIdx,2), 'filled', 'r');
                plot(...
                    [ratPosition(pIdx,1), ratPosition(pIdx,1) + fromRatToRobot(pIdx,1)],...
                    [ratPosition(pIdx,2), ratPosition(pIdx,2) + fromRatToRobot(pIdx,2)], '--');
                title(string(duration(seconds(timestamp_event(pIdx)), 'Format', 'mm:ss')));
            end

            if btwDistance(pIdx) >= 1
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

%% Rearrange
out_velocity_prism = [out_velocity(G1,1), out_velocity(G2,1)];
out_center_time_prism = [out_center_time(G1,1), out_center_time(G2,1)];
out_corner_time_prism = [out_corner_time(G1,1), out_corner_time(G2,1)];
out_mean_btw_distance_prism = [out_mean_btw_distance(G1,1), out_mean_btw_distance(G2,1)];
out_freezing_prism = [out_freezing(G1,1), out_freezing(G2,1)];
out_view_ratio_prism = [out_view_ratio(G1,1), out_view_ratio(G2,1)];
out_around_robot_ratio_prism = [out_around_robot_ratio(G1,1), out_around_robot_ratio(G2,1)];

%% btw distance histogram
btwD_def = cell2mat(histo_box(G1));
btwD_inf = cell2mat(histo_box(G2));

loadxkcd;
figure('Position', [-1474, 77, 628, 196]);
histogram(btwD_def,0:4:200, 'DisplayStyle', 'stairs', 'EdgeColor', xkcd.bright_red, 'LineWidth',1);
hold on;
histogram(btwD_inf,0:4:200, 'DisplayStyle', 'stairs', 'EdgeColor', xkcd.dark_red, 'LineWidth',1);
legend({"Deflated", "Inflated"}, 'Location', 'northwest')
xlabel('Distance (cm)');
ylabel('Count');
title('Head');
ylim([0, 1000]);
set(gca, 'FontName', 'Noto Sans');

figure('Position', [-1474, 77, 416, 196]);
clf;
histogram(btwD_def,0:4:200, 'DisplayStyle', 'stairs', 'EdgeColor', xkcd.bright_blue, 'LineWidth',1);
hold on;
histogram(btwD_inf,0:4:200, 'DisplayStyle', 'stairs', 'EdgeColor', xkcd.dark_blue, 'LineWidth',1);
legend({"Deflated", "Inflated"})
xlabel('Distance (cm)');
ylabel('Count');
title('Body');
ylim([0, 1000]);
set(gca, 'FontName', 'Noto Sans');