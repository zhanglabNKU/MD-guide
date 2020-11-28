%%%-------------------------------------------------------------------%%%
%%% Description:
%%% MATLAB implementation of the Medical Image Fusion (MIF)
%%% scheme described in "A Neuro-Fuzzy Approach for Medical Image Fusion" -
%%% IEEE-TBME - 2013.
%%% Author:
%%% Sudeb Das
%%% NB:
%%% 1. Input image type is Gray
%%% 2.
%%% Copyright:
%%% Dr. Sudeb Das, Prof. Malay Kumar Kundu, India
%%% References:
%%% If you use the code, please cite the paper as follows:
%%% Sudeb Das and Malay Kumar Kundu, "A Neuro-Fuzzy Approach for Medical
%%% Image Fusion?, IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING (IEEE-TBME),
%%% VOL. 60, NO. 12, 3347-3353, 2013.
%%%-------------------------------------------------------------------%%%
%% Clearing the workspace, closing all opened images, clearing the terminal
clear all, close all, clc
% Insert the path of the NSCT toolbox
% NSCT MATLAB toolbox can be found at:
% http://in.mathworks.com/matlabcentral/fileexchange/10049-nonsubsampled-contourlet-toolbox
% Compile the mex files according to your OS.
path(path,'nsct_toolbox/');

%% Inputing source images
% --------------------------------------------------------------------------
% % Source image A
% [filename1, pathname1] = uigetfile('*.*', 'Pick any image file');
% imA=imread(strcat(pathname1,filename1));
% % % Source image B
% [filename2, pathname2] = uigetfile('*.*', 'Pick any image file');
% imB=imread(strcat(pathname2,filename2));
%%以前的融合方式
% filename = 'F:\Medical-Image-Fusion-master\final_all_data.txt';
% [im1,im2]=textread(filename,'%s%s');
 
% path = 'F:\pet合成\论文2\20200714\IFCNN-master\Code\datasets\MDDataset\';
% filename = 'F:\pet合成\论文2\20200714\IFCNN-master\Code\datasets\MDDataset\test_list.txt';
% [im1,im2]=textread(filename,'%s%s');
% for k=1:size(im1);
    
 pet_path = 'F:\pet合成\论文2\20200714\outputs\medical\pet\';
 mr_path = 'F:\pet合成\论文2\20200714\outputs\medical\mr2\';
 fileExt = '*.png';
 files = dir(fullfile(pet_path,fileExt)); 
 files_mr =  dir(fullfile(mr_path,fileExt)); 
 len1 = size(files,1);
 for k=1:len1;
    
%     str=strrep(im1(k),'./data','F:\Medical-Image-Fusion-master\data');
%     str=strrep(str,'/','\');              
%     str2=strrep(str,'ct1','mr2');    
%     [I,img_ct]  = imread(char(str));
%     [I2,img_mr]  = imread(char(str2));
%     imA=ind2gray(I,img_ct);
%     imB=ind2gray(I2,img_mr);
imA  = imread(strcat(pet_path,char(files(k).name)));   %anatomical image
imB  = imread(strcat(mr_path,char(files_mr(k).name))); %functional image

%      str=strcat(path,char(im1(k)));
%      str2=strcat(path,char(im2(k)));
%      imA= imread(char(str));
%      imB = imread(char(str2));

% filename1 = 'a1.bmp';
% filename2 = 'b1.bmp';
% imA = imread(filename1);
% imB = imread(filename2);
%% Find the filenames and extensions of the source images considering there
% is only 1 '.' in the filename
%     dotPos1 = strfind(char(str),'.');
%     dotPos2 = strfind(char(str2),'.');
%     extName1 = filename1(dotPos1+1:end);
%     extName2 = filename2(dotPos2+1:end);
%     fName1 = filename1(1:dotPos1-1);
%     fName2 = filename2(1:dotPos2-1);

    %% Computation start from here
    [r1, c1, ch1] = size(imA);
    [r2, c2, ch2] = size(imB);

    % Source images' dimensions must be same and they should be co-registered
    % Convert from RGB to GRAY
    if(ch1>1)
        imA=rgb2gray(imA);
    end
    if(ch2>1)
        imB=rgb2gray(imB);
    end

    %% Parameter settings: can be modified to different values
    % % Parameters for PCNN
    Para.iterTimes=200;
    Para.link_arrange=3;
    Para.alpha_L=0.06931; % 0.06931 Or 1
    Para.alpha_Theta=0.2;
    Para.beta=0.2;        % 0.2 or 3 it will be set adaptively 
    Para.vL=1;
    Para.vTheta=20;

    Para.W =computeGauusianWeightWindow(Para.link_arrange);

    % Fileter size
    filter_size = 3;

    % % NSCT parameters
    NSCTPara.levels = [1,2,2];        % Decomposition level
    NSCTPara.pfilt = 'pyrexc';        % Pyramidal filter
    NSCTPara.dfilt = 'vk' ;           % Directional filter

    % % Main fusion function
    imF  = nsctFusion(imA, imB, filter_size, Para, NSCTPara);

    %% Displaying different statistical measures
    sourceImage1Entropy = entropy(imA);
    sourceImage2Entropy = entropy(imB);
    fusedImageEntropy   = entropy(imF);
    sourceImage1stadardDeviation = std2(imA);
    sourceImage2stadardDeviation = std2(imB);
    fusedImagestadardDeviation = std2(imF);

    disp(strcat('Source image 1 entropy = ',num2str(sourceImage1Entropy)));
    disp(strcat('Source image 2 entropy = ',num2str(sourceImage2Entropy)));
    disp(strcat('Fused  image   entropy = ',num2str(fusedImageEntropy)));
    disp(strcat('Source image 1 standard deviation = ',num2str(sourceImage1stadardDeviation)));
    disp(strcat('Source image 2 standard deviation = ',num2str(sourceImage2stadardDeviation)));
    disp(strcat('Fused  image   standard deviation = ',num2str(fusedImagestadardDeviation)));
    
%     fusion_name=strrep(im1(k),'gif','tif'); 
%     fusion_name=strrep(fusion_name,'/','_'); 
    fusion_name=strrep(fusion_name,'._data','./results/');

    fusion_name=strrep(im1(k),'_1','_result'); 
    fusion_name=strrep(fusion_name,'c','./result/c'); 

%     imwrite(uint8(imF),char(fusion_name));
    subplot(1,3,1), imshow(imA), title('Source Image 1')
    subplot(1,3,2), imshow(imB), title('Source Image 2')
    subplot(1,3,3), imshow(imF), title('Fused Image')
end
