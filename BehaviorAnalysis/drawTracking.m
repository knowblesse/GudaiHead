function drawTracking(targetdir)
%%
if exist("targetdir", "var")
    targetdir = uigetdir(targetdir);
else
    targetdir = uigetdir();
end

load('apparatus.mat');

trackingFilePath = fullfile(targetdir, "tracking.csv");
data = readmatrix(trackingFilePath, "Delimiter",',');

%% 

data_interp = [...
    data(:,1),...
    interp1(1:size(data,1), data(:,2), 1:size(data,1), 'spline')',...
    interp1(1:size(data,1), data(:,3), 1:size(data,1), 'spline')',...
    interp1(1:size(data,1), data(:,4), 1:size(data,1), 'spline')',...
    interp1(1:size(data,1), data(:,5), 1:size(data,1), 'spline')'
    ];


%% Plot Track
fig = figure();
clf;
ax = axes;
ax.YDir = 'reverse';
apparatusImage = apparatus.image;
image(apparatusImage, "AlphaData",0.5 * ones(size(apparatusImage,1), size(apparatusImage,2)));
hold on;

range = 1 : size(data,1);
%range = 1 : 1000;

plot(data(range,2), data(range,3), 'Color', 'r', 'LineWidth',1);
plot(data(range,4), data(range,5), 'Color', 'g', 'LineWidth',1);

end

