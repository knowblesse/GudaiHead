%%
addpath('..');

for session = 1 : 10
    FPS = 60;
    %folderPath = 'D:\Data_fib\Robot Predator\R9';
    folderPath = strcat('D:\Data_fib\Robot Predator\Rtest2\R', num2str(session, '%02d'));
    timeData = readlines(glob(folderPath, '.*.txt', true));
    timeData = timeData(1 : 5);
    trackingData = readmatrix(glob(folderPath, '.*.csv', true));
    
    timestamp = trackingData(:,1) / FPS;
    
    separator = seconds(duration(timeData, 'InputFormat', 'mm:ss'));
    timestampIndex = [1; arrayfun(@(x) find(timestamp>x, 1), separator)];
    
    titles = {"Hab", "Robot no Head", "Rest1", "Robot with Head", "Rest2"};
    
    fig = figure(1);
    clf;
    for i = 1 : 5
        subplot(1,5,i);                                         
        hold on;
        plot(...
            trackingData(timestampIndex(i):timestampIndex(i+1), 2),...
            trackingData(timestampIndex(i):timestampIndex(i+1), 3),'r');
        plot(...
            trackingData(timestampIndex(i):timestampIndex(i+1), 4),...
            trackingData(timestampIndex(i):timestampIndex(i+1), 5),'g');
        xlim([0, 1600]);
        ylim([0, 1000]);
        title(titles{i});
    end
    saveas(fig, strcat('Image/R', num2str(session)), 'png');
end
% disp(duration(seconds(timestamp(end)), 'Format', 'mm:ss'));
