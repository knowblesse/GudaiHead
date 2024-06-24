vr = VideoReader('E:\Data_fib\Robot Predator\Two_rat_playing_1.mkv');
image = vr.read(1);

height = size(image, 1);

%%
figure(1);
clf;
imshow(image);
hold on;

plot(raw(:,3), height - raw(:, 4));

%%
load('apparatus.mat');
ratPosition = raw(1:36001,3:4);
fps = 60;
px2cm = 0.2171;

% Velocity
deltaPosition = sum(diff(ratPosition).^2,2).^0.5;
bgNumber = 2;
velocity = zeros(10,1);
center_ratio = zeros(10, 1);
for m = 1 : 10
    velocity(m) = sum(deltaPosition( 3600*(m-1)+1: 3600*m)) / 60 * px2cm;
    inCenter = 0;
    for index = 3600*(m-1)+1 : 3600*m
        inCenter = inCenter + apparatus.center(...
            height - round(ratPosition(index, 2)),...
            round(ratPosition(index, 1)),... 
            bgNumber);
    end
    center_ratio(m) = inCenter / 3600;
end
