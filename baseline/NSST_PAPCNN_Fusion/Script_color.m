clear all;
close all;
clc;
%% NSST tool box
addpath(genpath('shearlet'));
%%
 pet_path = 'F:\pet合成\论文2\20200714\outputs\medical\pet\';
 mr_path = 'F:\pet合成\论文2\20200714\outputs\medical\mr2\';
 fileExt = '*.png';
 files = dir(fullfile(pet_path,fileExt)); 
 files_mr =  dir(fullfile(mr_path,fileExt)); 
 len1 = size(files,1);
 for k=1:len1;
B  = imread(strcat(pet_path,char(files(k).name)));   %anatomical image
A  = imread(strcat(mr_path,char(files_mr(k).name))); %functional image
% A=imread('sourceimages/s02_MR.tif');  %anatomical image
% B=imread('sourceimages/s02_PET.tif'); %functional image
% figure;imshow(A);
% figure;imshow(B);

img1 = double(A)/255;
img2 = double(B)/255;
img2_YUV=ConvertRGBtoYUV(img2);
img2_Y=img2_YUV(:,:,1);
[hei, wid] = size(img1);

% image fusion with NSST-PAPCNN 
imgf_Y=fuse_NSST_PAPCNN(img1,img2_Y); 

imgf_YUV=zeros(hei,wid,3);
imgf_YUV(:,:,1)=imgf_Y;
imgf_YUV(:,:,2)=img2_YUV(:,:,2);
imgf_YUV(:,:,3)=img2_YUV(:,:,3);
imgf=ConvertYUVtoRGB(imgf_YUV);

F=uint8(imgf*255);
figure,imshow(F);
imwrite(F,strcat('F:\文章1\Medical-Image-Fusion-master\baseline\liuYu\NSST_PAPCNN_Fusion\result\pet-mr\',files(k).name(end-6:end-4),'.png'));
  end

