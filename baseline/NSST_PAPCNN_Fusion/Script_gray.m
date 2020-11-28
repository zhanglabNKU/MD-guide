clear all;
close all;
clc;
%% NSST tool box
addpath(genpath('shearlet'));
%%
% filename = 'F:\Medical-Image-Fusion-master\final_all_data.txt';
%  [im1,im2]=textread(filename,'%s%s');
path = 'F:\pet合成\论文2\20200714\IFCNN-master\Code\datasets\MDDataset\';
filename = 'F:\pet合成\论文2\20200714\IFCNN-master\Code\datasets\MDDataset\test_list.txt';
[im1,im2]=textread(filename,'%s%s');
 
for k=1:size(im1);
%     str=strrep(im1(k),'./data','F:\Medical-Image-Fusion-master\data');
%     str=strrep(str,'/','\');              
%     str2=strrep(str,'ct1','mr2');
%      
%     [I,img_ct]  = imread(char(str));
%     [I2,img_mr]  = imread(char(str2));
%     A=ind2gray(I,img_ct);
%     B=ind2gray(I2,img_mr);
    
    str=strcat(path,char(im1(k)));
    str2=strcat(path,char(im2(k)));
    A = imread(char(str));
    B = imread(char(str2));
    
    
    tic;
%     A=imread('sourceimages/s01_CT.tif');
%     B=imread('sourceimages/s01_MRT2.tif');
%     figure;imshow(A);
%     figure;imshow(B);

    img1 = double(A)/255;
    img2 = double(B)/255;

    % image fusion with NSST-PAPCNN 
    imgf=fuse_NSST_PAPCNN(img1,img2); 

    F=uint8(imgf*255);
%     figure,imshow(F);
    
%     fusion_name=strrep(im1(k),'gif','tif'); 
%     fusion_name=strrep(fusion_name,'/','_'); 
%     fusion_name=strrep(fusion_name,'._data','./results/');
    fusion_name=strrep(im1(k),'_1','_result'); 
    fusion_name=strrep(fusion_name,'c','./result/c'); 
    
    imwrite(F,char(fusion_name));
    toc;
end