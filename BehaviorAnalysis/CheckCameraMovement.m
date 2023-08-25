%% CheckCameraMovement

%% Constants
BASEPATH = 'D:\Data_fib\Robot Predator\Rtest1';


%% Draw frames for background moving
frames = [];

for session = 1 : 10
    folderPath = fullfile(BASEPATH, strcat('R', num2str(session,'%02d')));
    vr = VideoReader(glob(folderPath, '.*mkv', true));
    fr = vr.readFrame();
    frames = cat(4, frames, fr);
end
        
figure(1);
clf;
for i = 1 : 10
    imshow(frames(:,:,:,i));
    pause(0.5);
end

%% Load bg.png and save Border
bg = false([size(frame,[1,2]),7]);

for i = 1 : 7

    frame = imread(strcat('bg',num2str(i),'.png'));
    clf;
    imshow(255-frame);
    border = round(ginput(2)); % top left corner and bot right corner
    
    mask = false(size(frame,[1,2]));
    
    mask(...
        border(1,2)+70:border(2,2)-70,...
        border(1,1)+70:border(2,1)-70) = true;
    
    
    clf;
    imshow(frame);
    hold on;
    
    overlay = uint8(zeros(size(frame)));
    overlay(:,:,1) = uint8(mask) * 255;
    
    
    a = imshow(overlay);
    a.AlphaData = 0.2;
    
    bg(:,:,i) = mask;
end

%% Check
clf;
for i = 1 : 7
    imshow(frame);
    hold on;
    overlay = uint8(zeros(size(frame)));
    overlay(:,:,1) = uint8(bg(:,:,i)) * 255;
    a = imshow(overlay);
    a.AlphaData = 0.2;
    drawnow;
    pause(0.5);
end

%% Using apparatus.edge, generate apparatus.corner
corner = false([size(apparatus.center,[1,2]),7]);
for i = 1 : 7    
    row_start = find(apparatus.center(:,round(size(apparatus.center,2)/2),i) == 1, 1);
    row_end = find(apparatus.center(:,round(size(apparatus.center,2)/2),i) == 1, 1, 'last');
    col_start = find(apparatus.center(round(size(apparatus.center,1)/2),:,i) == 1, 1);
    col_end = find(apparatus.center(round(size(apparatus.center,1)/2),:,i) == 1, 1, 'last');

    corner(row_start-70:row_start+70, col_start-70:col_start+70, i) = true;
    corner(row_end-70:row_end+70, col_start-70:col_start+70, i) = true;
    corner(row_start-70:row_start+70, col_end-70:col_end+70, i) = true;
    corner(row_end-70:row_end+70, col_end-70:col_end+70, i) = true;
end

apparatus.corner = corner;

corner = false([size(apparatus.center,[1,2]),7]);
for i = 1 : 7    
    row_start = find(apparatus.center(:,round(size(apparatus.center,2)/2),i) == 1, 1);
    row_end = find(apparatus.center(:,round(size(apparatus.center,2)/2),i) == 1, 1, 'last');
    col_start = find(apparatus.center(round(size(apparatus.center,1)/2),:,i) == 1, 1);
    col_end = find(apparatus.center(round(size(apparatus.center,1)/2),:,i) == 1, 1, 'last');

    corner(row_start-70:row_start, col_start-70:col_start, i) = true;
    corner(row_end:row_end+70, col_start-70:col_start, i) = true;
    corner(row_start-70:row_start, col_end:col_end+70, i) = true;
    corner(row_end:row_end+70, col_end:col_end+70, i) = true;

%     corner(row_start-70:row_start+70, col_start-70:col_start, i) = true;
%     corner(row_end-70:row_end+70, col_start-70:col_start, i) = true;
%     corner(row_start-70:row_start+70, col_end:col_end+70, i) = true;
%     corner(row_end-70:row_end+70, col_end:col_end+70, i) = true;
% 
%     corner(row_start-70:row_start, col_start:col_start+70, i) = true;
%     corner(row_end:row_end+70, col_start:col_start+70, i) = true;
%     corner(row_start-70:row_start, col_end-70:col_end, i) = true;
%     corner(row_end:row_end+70, col_end-70:col_end, i) = true;

end

apparatus.corner = corner;
figure();
for i = 1 : 7
    clf;
    imshow(apparatus.image(:,:,:,i));
    hold on;
    a = imshow(apparatus.center(:,:,i));
    b = imshow(apparatus.corner(:,:,i));
    a.AlphaData = 0.2;
    b.AlphaData = 0.2;
    pause(1);
end

    

