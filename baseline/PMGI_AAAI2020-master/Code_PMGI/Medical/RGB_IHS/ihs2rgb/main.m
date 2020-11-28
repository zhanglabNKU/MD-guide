clc;
clear all;

SamplePaths = '/home/hww/fusion0820/baseline/PMGI_AAAI2020-master/Code_PMGI/Medical/pet-mri/fusion/';
fileExt = '*.png';
files = dir(fullfile(SamplePaths,fileExt))

len = size(files,1);

for i = 1:len
    fileName = strcat(SamplePaths,files(i).name);

%         str = strrep(fileName,'/','\');        
    I_result=im2double(imread(fileName)); 
   
    I_init=im2double(imread(strcat('/home/hww/fusion0820/MF_dataset/Code/rgb_data/case22/dg1__enlarge_',files(i).name)));
    [I,V1,V2]=rgb2ihs(I_init);
    I_final_ISH=cat(3,I_result,V1,V2);
    I_final_RGB=ihs2rgb(I_final_ISH);
    
    imwrite(I_final_RGB, strcat('',fileName(end-6:end-4),'.bmp')); 
    
      end 


% for num=1:70
%   for i=1:19
%       if i<=10
%          I_result=im2double(imread(strcat('',num2str(num),'\','F9_0',num2str(i-1),'.bmp')));  
%       else
%          I_result=im2double(imread(strcat('',num2str(num),'\','F9_',num2str(i-1),'.bmp')));
%       end
%       I_init=im2double(imread(strcat('',num2str(i),'.bmp')));
%       [I,V1,V2]=rgb2ihs(I_init);
%       I_final_ISH=cat(3,I_result,V1,V2);
%       I_final_RGB=ihs2rgb(I_final_ISH);
%       if ~exist(strcat('',num2str(num))) ;
%          mkdir(strcat('',num2str(num)));
%       end
%       if i<=10
%          imwrite(I_final_RGB, strcat('',num2str(num),'\','F9_0',num2str(i-1),'.bmp')); 
%       else
%          imwrite(I_final_RGB, strcat('',num2str(num),'\','F9_',num2str(i-1),'.bmp')); 
%       end
%     end
% end