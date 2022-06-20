%*************************************************************************%
%                                                                         %
%       Date:               10/06/2022                                    %
%       Author:             Group 1 (PBL S2)                              %
%       Main description:   3D reconstruction of a rabbit                 %
%       Version:            1.0                                           %
%                                                                         %
%*************************************************************************%
clc;
clear all;
close all;

tic;
%% Optional parameters
fprintf('[INFO] Running point cloud generation based on stereovision...\n\n');

% Cloud point generation
WriteCloudPoint = 1;

% Ploting option
PlotMatches = 0;
PlotMaxDisp = 0;
PlotMatchedRect = 0;
PlotOrigRect = 0;
PlotDispMap = 0;
PlotDepthMap = 0;
PlotCloudPoint = 0;

% Method to extract features
% Method = 'Harris';
Method = 'SURF';
% Method = 'SIFT';

%% Load components folder
folder = fileparts(which(mfilename));
addpath(genpath(folder));

nImg = length(dir([strcat(folder,'\img\rabbit') '/*.bmp']));

if ~(exist('results','file'))
    mkdir('results');
end

%% Data
Targets = [[ 243.373657, -352.378204, 436.373932, 57.2547569, -35.9195290, -172.683960];
        [ 293.511047, -378.905060, 417.162445, 65.4235382, -38.2280350, -171.148087];
        [ 337.884674, -378.611298, 417.162445, 69.3516617, -38.2757, -171.120316];
        [ 381.703857, -378.611298, 417.162445, 80.5260849, -38.0873070, -171.138046];
        [ 453.902222, -378.602112, 417.160675, 90.4676056, -38.0937119, -171.132584];
        [ 525.000305, -378.605194, 417.162415, 102.424477, -38.0936, -171.132797];
        [ 562.769653, -313.659027, 410.420532, 111.539497, -38.0937347, -171.132965];
        [ 608.734497, -232.922348, 406.612213, 132.062119, -38.4224548, -170.909500];
        [ 642.551147, -109.864288, 406.537292, 154.912735, -37.1213608, -171.955460];
        [ 651.614258, -18.4677143, 406.537292, 174.861, -37.1213608, -171.955460];
        [ 654.317749, 34.0319176, 406.488678, -173.663696, -37.1159897, -171.946503];
        [ 648.622681, 92.6202545, 406.488678, -162.528336, -37.1159897, -171.946503];
        [ 634.545288, 163.224243, 406.488678, -149.104080, -37.1159897, -171.946503];
        [ 593.362122, 277.869171, 423.103394, -132.516663, -37.1159897, -171.946503];
        [ 531.373901, 310.826721, 423.110199, -119.315735, -37.1201859, -171.955795];
        [ 506.978241, 311.805908, 423.110199, -114.664604, -37.1201859, -171.955795];
        [ 461.574402, 300.137787, 425.145844, -107.109367, -37.1201210, -171.956055];
        [ 381.229767, 300.136475, 419.155090, -94.3278, -37.1201477, -171.956039];
        [ 295.845276, 290.233154, 419.155090, -80.2537, -37.1201477, -171.956039];
        [ 235.127670, 290.233154, 419.155090, -70.2442627, -37.1201477, -171.956039];
        [ 189.504868, 205.631287, 419.155090, -57.2024956, -37.1201477, -171.956039];
        [ 141.840714, 173.705978, 419.153412, -48.9320297, -37.1201973, -171.956268];
        [ 141.839523, 173.704132, 522.805420, -31.9285107, -23.2908363, 172.932419];
        [ 193.744400, 207.280701, 522.805481, -37.6639137, -21.0465374, 171.037857];
        [ 267.247772, 239.833344, 522.805481, -49.5912285, -18.0864277, 167.265945];
        [ 335.276398, 260.626953, 522.805481, -60.7216072, -18.0864277, 167.265945];
        [ 407.953400, 260.625854, 522.804810, -54.1604462, -8.87295914, 147.842682];
        [ 478.602051, 283.130951, 522.804382, -70.9105911, -10.4173765, 147.301941];
        [ 525.052917, 207.962, 522.805176, -84.6371231, -9.32944775, 153.227386];
        [ 565.934875, 166.003326, 519.130310, -99.7657318, -9.32944775, 153.227386];
        [ 620.558, 151.139603, 519.136902, -107.201393, -9.32106876, 153.234650];
        [ 632.930969, 85.6372604, 508.591766, -120.767670, -9.81905937, 152.028381];
        [ 643.197876, 37.1157494, 508.565552, -131.625946, -9.77595901, 152.008072];
        [ 655.665649, -4.83302927, 508.529053, -140.076920, -9.77483273, 151.987640];
        [ 647.239075, -181.820450, 508.194763, 168.669373, -16.9013882, 160.954987];
        [ 620.681091, -258.848663, 508.064148, 157.974686, -20.0942268, 152.364090];
        [ 579.883789, -340.557373, 507.678619, 150.187698, -20.0942097, 152.364197];
        [ 535.229431, -347.190674, 507.589661, 143.437897, -20.0225372, 152.327972];
        [ 480.391846, -305.834442, 507.589, 145.402924, -9.69712257, 144.707275];
        [ 373.488800, -297.644562, 507.570984, 119.118378, -11.1686106, 149.270615];
        [ 279.762360, -297.644562, 507.570984, 99.9958344, -11.8950806, 150.609];
        [ 209.630051, -297.644562, 526.911804, 86.4089127, -11.7164345, 149.584503];
        [ 185.659363, -224.841934, 526.911316, 73.9275208, -11.6877518, 150.045303];
        [ 163.852890, -175.711029, 526.902161, 61.2171631, -11.6879568, 150.044601];
        [ 163.870285, -90.3245773, 526.903, 26.6666985, -14.6855659, 169.935486];
        [ 163.871368, -90.3252106, 601.519714, 22.4130669, -5.51762247, 173.836014];
        [ 163.881607, 64.5824356, 601.520630, -11.0707073, -2.56019950, 173.968597];
        [ 433.759888, 61.2725372, 601.555786, -13.6851969, 34.6189919, 160.405563]];

n = length(Targets);

load('cameraParams');
load('handEye');

%% Loop
for id1 = 1 : nImg - 1
    id2 = id1 + 1;

    % Load images
    if (id1 < 10)
        id1_str = strcat('0',num2str(id1)); 
    else
        id1_str = strcat(num2str(id1));
    end
    if (id2 < 10)
        id2_str = strcat('0',num2str(id2));
    else
        id2_str = strcat(num2str(id2));
    end
    im1 = imread(sprintf('img/rabbit/RAB%s.bmp',id1_str));
    im2 = imread(sprintf('img/rabbit/RAB%s.bmp',id2_str));
    
    % Convert images to grayscale
    img1_dist = rgb2gray(im1);
    img2_dist = rgb2gray(im2);
    
    % Undistort images
    im1_undist = undistortImage(im1, cameraParams);
    im2_undist = undistortImage(im2, cameraParams);
    img1 = undistortImage(img1_dist, cameraParams);
    img2 = undistortImage(img2_dist, cameraParams);
    
    % Extract features
    switch (Method)
        case 'Harris'
            points1 = detectHarrisFeatures(img1);
            points2 = detectHarrisFeatures(img2);
    
        case 'SURF'
            points1 = detectSURFFeatures(img1);
            points2 = detectSURFFeatures(img2);
       
        otherwise % case 'SIFT'
            points1 = detectSIFTFeatures(img1);
            points2 = detectSIFTFeatures(img2);
    end
    [features1, valid_points1] = extractFeatures(img1, points1);
    [features2, valid_points2] = extractFeatures(img2, points2);
    
    % Matched points
    indexPairs = matchFeatures(features1, features2);
    
    matchedPoints1 = valid_points1(indexPairs(:,1),:);
    matchedPoints2 = valid_points2(indexPairs(:,2),:);
    
    % Fundamental matrix
    [F, inliersIndex_F] = estimateFundamentalMatrix(matchedPoints1, matchedPoints2);
    
    % Essential matrix
    [E, inliersIndex_E] = estimateEssentialMatrix(matchedPoints1, matchedPoints2, cameraParams);
    
    % Inlier points
    % NOTE. The inlier points are extracted from the matched points (not the
    % valid points because there are outliers).
    inlierPoints1 = matchedPoints1(inliersIndex_E);
    inlierPoints2 = matchedPoints2(inliersIndex_E);
    
    % Plot example
    if (PlotMatches)
        id_example = 5;
        matchedPoints1_xy = matchedPoints1(id_example,:).Location;
        matchedPoints2_xy = matchedPoints2(id_example,:).Location;

        figure;
        imshow(img1);
        hold on;
        plot(matchedPoints1_xy(1), matchedPoints1_xy(2), 'o', 'LineWidth', 4);
        
        figure;
        imshow(img2);
        hold on;
        plot(matchedPoints2_xy(1), matchedPoints2_xy(2), 'o', 'LineWidth', 4);
        
        figure;
        showMatchedFeatures(img1, img2, matchedPoints1(1:20), matchedPoints2(1:20));

        figure;
        showMatchedFeatures(img1, img2, matchedPoints1(1:20), matchedPoints2(1:20),...
            'montage', 'PlotOptions', {'ro','go','y--'});
    end
    
    %% Fundamental and essential matrix verification
    % Obtain relative camera pose
    [relativeOrientation, relativeLocation] = relativeCameraPose(E, ...
        cameraParams, inlierPoints1, inlierPoints2);
    
    % Rotations from world to camera 1/camera 2
    R1 = euler_rot(Targets(id1, 4:end));
    R2 = euler_rot(Targets(id2, 4:end));
    
    % Translations from world to camera 1/camera 2
    T1 = Targets(id1, 1:3)';
    T2 = Targets(id2, 1:3)';
    
    % Transformation from camera 2 to camera 1
    w_R1T1 = [[R1, T1]; [0 0 0 1]]; % world to gripper of camera 1
    w_R2T2 = [[R2, T2]; [0 0 0 1]]; % world to gripper of camera 2
    
    w_RT_cam1 = w_R1T1 * [Rx, Tx; 0 0 0 1]; % world to camera 1
    w_RT_cam2 = w_R2T2 * [Rx, Tx; 0 0 0 1]; % world to camera 2
    
    cam1_RT_cam2 = pinv(w_RT_cam1) * w_RT_cam2; % camera 2 to camera 1
    R = cam1_RT_cam2(1:3, 1:3);
    T = cam1_RT_cam2(1:3,4);
    
    % Rotation error
    % NOTE. In MATLAB the rotations are transposed.
    error_rot = relativeOrientation' - R;
    
    % Check translation direction
    trans_diff = (relativeLocation' - T/norm(T));
    
    % Angle between both directions
    angle_diff = acos(relativeLocation * (T / norm(T))); % [rad]
    
    if (abs(angle_diff) < 0.05) % error max 0.05 radians
        trans_info = 'The translation vectors have the same direction\n\n';
    else
        trans_info = 'The translation vectors do NOT have the same direction\n\n';
    end
    
    % Check the fundamental and essential matrices
    A = cameraParams.IntrinsicMatrix';
    
    E_theo = R' * myskew(T);
    F_theo = inv(A') * E_theo * inv(A);

    E_theo = - E_theo / norm(E_theo); % The sign of E, F can be the opposite (as it is)
    
    error_E = E_theo - E;
    
    F_theo = - F_theo / norm(F_theo);
    error_F = F_theo - F;
    
    %% Rectification
    % Use R/T obtained from "relativeCameraPose" function
    R = relativeOrientation';
    T = relativeLocation' * norm(T);
    
    % Define homographies
    Xcr = T/norm(T);
    Ycr = cross([0, 0, 1]', Xcr)/norm(cross([0, 0, 1]', Xcr));
    Zcr = cross(Xcr, Ycr);
    Rect1 = [Xcr'; Ycr'; Zcr'];
    H1 = A * Rect1 * inv(A);
    H1 = H1 / H1(3, 3);
    H2 = A * Rect1 * R * inv(A);
    H2 = H2 / H2(3, 3);
    
    % Calculate with H1 the max points in the image
    H1_00 = H1 * [0; 0; 1]; H1_00 = H1_00 / H1_00(3);
    H1_W0 = H1 * [size(img1, 2); 0; 1]; H1_W0 = H1_W0 / H1_W0(3);
    H1_0H = H1 * [0; size(img1, 1); 1]; H1_0H = H1_0H / H1_0H(3);
    H1_WH = H1 * [size(img1, 2); size(img1, 1); 1]; H1_WH = H1_WH / H1_WH(3);
    maxpts1H1Y = max([H1_00(2), H1_W0(2), H1_0H(2), H1_WH(2)]);
    minpts1H1Y = min([H1_00(2), H1_W0(2), H1_0H(2), H1_WH(2)]);
    maxpts1H1X = max([H1_00(1), H1_W0(1), H1_0H(1), H1_WH(1)]);
    minpts1H1X = min([H1_00(1), H1_W0(1), H1_0H(1), H1_WH(1)]);
    
    maxDisp = 0;
    Diffs = zeros(1, length(inlierPoints1));
    for i = 1 : length(inlierPoints1)
      Pi1 = inlierPoints1(i).Location;
      Pi2 = inlierPoints2(i).Location;
      HPi1 = H1 * [Pi1(1); Pi1(2); 1];
      HPi1 = HPi1 / HPi1(3);
      HPi2 = H2 * [Pi2(1); Pi2(2); 1];
      HPi2 = HPi2 / HPi2(3);
      Diffs(i) = (HPi1(1) - HPi2(1));
      maxDisp = max(maxDisp, Diffs(i));
    end
    
    % Plot the point with maximum disparity
    idx_maxDisp = find(Diffs == maxDisp);
    iP1_maxDisp = inlierPoints1.Location(idx_maxDisp,:);
    iP2_maxDisp = inlierPoints2.Location(idx_maxDisp,:);
    if (PlotMaxDisp)
        figure;
        showMatchedFeatures(img1,img2,iP1_maxDisp,iP2_maxDisp, ...
            'montage', 'PlotOptions', {'ro','go','y--'});
        %title('Maximum difference point');
    end
    
    % Limit the range of disparity to [0 256]
    if (maxDisp < 256)
      DiffDisp = 0;
    else
      DiffDisp = maxDisp + 25 - 256;
    end
    
    % Define references for rectification
    Ref1 = imref2d([int32(maxpts1H1Y - minpts1H1Y), int32(maxpts1H1X - minpts1H1X)], ...
                     [minpts1H1X, maxpts1H1X], [minpts1H1Y, maxpts1H1Y]);
    Ref2 = imref2d([int32(maxpts1H1Y - minpts1H1Y), int32(maxpts1H1X - minpts1H1X)], ...
                     [minpts1H1X - DiffDisp, maxpts1H1X - DiffDisp], [minpts1H1Y, maxpts1H1Y]);
            
    DiffX = maxpts1H1X - minpts1H1X;
    DiffY = maxpts1H1Y - minpts1H1Y;

    % Rectify images
    img1_rect = imwarp(img1, projective2d(H1'), 'OutputView', Ref1, 'FillValues', 0); % grayscale
    img2_rect = imwarp(img2, projective2d(H2'), 'OutputView', Ref2, 'FillValue', 0); % grayscale
    
    im1_rect = imwarp(im1_undist, projective2d(H1'), 'OutputView', Ref1, 'FillValues', 0); % RGB
    im2_rect = imwarp(im2_undist, projective2d(H2'), 'OutputView', Ref2, 'FillValues', 0); % RGB
    
    % Calculate the matched points in the rectified images
    switch (Method)
        case 'Harris'
            imagePoints1_rect = detectHarrisFeatures(img1_rect);
            imagePoints2_rect = detectHarrisFeatures(img2_rect);
    
        case 'SURF'
            imagePoints1_rect = detectSURFFeatures(img1_rect);
            imagePoints2_rect = detectSURFFeatures(img2_rect);
       
        otherwise % case 'SIFT'
            imagePoints1_rect = detectSIFTFeatures(img1_rect);
            imagePoints2_rect = detectSIFTFeatures(img2_rect);
    end
    
    features1_rect = extractFeatures(img1_rect, imagePoints1_rect, ...
                                        'Upright', true); %, 'FeatureSize', 128);
    features2_rect = extractFeatures(img2_rect, imagePoints2_rect, ...
                                        'Upright', true); %, 'FeatureSize', 128);
    
    indexPairs_rect = matchFeatures(features1_rect, features2_rect, ...
        'MaxRatio', 0.9, 'MatchThreshold', 90);
    matchedPoints1_rect = imagePoints1_rect(indexPairs_rect(:,1));
    matchedPoints2_rect = imagePoints2_rect(indexPairs_rect(:,2));
    
    [E_rect, inliersIdx_rect] = estimateEssentialMatrix(matchedPoints1_rect, ...
        matchedPoints2_rect, cameraParams);
    inlierPoints1_rect = matchedPoints1_rect(inliersIdx_rect);
    inlierPoints2_rect = matchedPoints2_rect(inliersIdx_rect);
    
    % Plot rectified images with matched points
    if (PlotMatchedRect)
        figure;
        showMatchedFeatures(img1_rect, img2_rect, inlierPoints1_rect(1:20), ...
            inlierPoints2_rect(1:20), 'montage', 'PlotOptions', {'ro','go','y--'});
        %title('Inlier Matches');
    end
    
    % Plot original and rectified images
    if (PlotOrigRect)
        figure;
        subplot(2,2,1); imshow(im1_undist, imref2d(size(img1)));
        subplot(2,2,2); imshow(im2_undist, imref2d(size(img2)));
        subplot(2,2,3); imshow(img1_rect, Ref1);
        subplot(2,2,4); imshow(img2_rect, Ref2);
    end
    
    %% Disparity calculation
    % Disparity
    % NOTE. If disparities are not found, NaN is returned.
    dispMap = disparityBM(img1_rect, img2_rect, 'DisparityRange', [0 256], ...
       'UniquenessThreshold', 16);

    % Filter disparity values to identify the rabbit
    Tth = 0.35 * 256;   % manual threshold based on disp map
    b = T(1); % relative translation in X-axis
    f = cameraParams.Intrinsics.FocalLength(1); % focal length of the camera (fx)

    [nRows, nCols] = size(dispMap);
    k = 1;
    disp = zeros(nRows, nCols);
    % x = zeros(nRows, nCols);
    % y = zeros(nRows, nCols);
    depthMap = zeros(nRows, nCols);
    for i = 1 : nRows
        for j = 1 : nCols
            if (dispMap(i,j) >= Tth)
                disp(i,j) = dispMap(i,j);

                x_array(k) = j; % column number
                y_array(k) = i; % row number

                disp_array(k) = disp(i,j);

                red(k) = double(im1_rect(i,j,1));
                green(k) = double(im1_rect(i,j,2));
                blue(k) = double(im1_rect(i,j,3));

                z_array(k) = b * f ./ disp_array(k); % not used
                depthMap(i,j) = z_array(k);

                k = k + 1;
            else
                disp(i,j) = 0;
                depthMap(i,j) = 0;
            end
        end
    end

    m1r = [x_array; y_array; z_array];

    % Plot disparity map
    if (PlotDispMap)
        figure;
        imshow(dispMap, []); % grayscale
        %title('Disparity map');

        figure;
        imshow(disp);
        %title('Disparity map corresponding to rabbit (manual thresholding)');
    end

    if (PlotDepthMap)
        figure;
        imshow(depthMap, []); % grayscale
        %title('Depth map');
    end
    
    % Q matrix
    Q = [1, 0, 0, - A(1, 3); ...
         0, 1, 0, - A(2, 3); ...
         0, 0, 0, A(1,1);...
         0, 0, 1/norm(T), 0];
    
    x1 = m1r(1,:) + DiffX;
    y1 = m1r(2,:) + DiffY;
    disp_array = disp_array + DiffDisp;
    ptsDisp = Q * [x1; y1; disp_array; ones(1, size(m1r,2))];
    ptsDisp = ptsDisp ./ ptsDisp(4, :); % normalization

    % Transform between world frame and rectified cam1 frame
    RcTc1 = [Rect1, [0;0;0]; 0 0 0 1] * w_RT_cam1;
    ptsOrg = inv(RcTc1) * ptsDisp;
    
    %% Cloud point generation
    % Save RGB values
    ptsOrg_RGB = [(ptsOrg(1:3,:))', red', green', blue'];
    
    % Write in .txt file
    if (WriteCloudPoint)
        % writematrix((ptsOrg(1:3,:))', 'PointCloud.txt');
        writematrix(ptsOrg_RGB, sprintf('results/PointCloud_%s_%s.txt', ...
            id1_str, id2_str));
    end
    
    % Plot cloud point in MATLAB
    if (PlotCloudPoint)
        figure;
        ptCloud = pointCloud((ptsOrg(1:3,:))');
        % ptCloud = pointCloud((ptsOrg_RGB(1:3,:))', 'Color', (ptsOrg_RGB(4:end,:))');
        pcshow(ptCloud);
    end
end

toc;
fprintf('[INFO] Point cloud generation based on sterevision has been completed!\n\n');

%% Functions
function s = myskew(q)
    if numel(q) ~= 3
         error('Input vector must have 3 elements.')
    end
    
    s = [0 -q(3) q(2); ...
        q(3) 0 -q(1); ...
        -q(2) q(1) 0];
end

function Rbg = euler_rot(euler_angles)
    A = euler_angles(1);
    B = euler_angles(2);
    C = euler_angles(3);

    Rx_C = [1 0 0; 0 cosd(C) -sind(C); 0 sind(C) cosd(C)];
    Ry_B = [cosd(B) 0 sind(B); 0 1 0; -sind(B) 0 cosd(B)];
    Rz_A = [cosd(A) -sind(A) 0; sind(A) cosd(A) 0; 0 0 1];
    
    Rbg = Rz_A * Ry_B * Rx_C;
end
