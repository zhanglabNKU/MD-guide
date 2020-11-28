clc;
clear all;

SamplePaths = '/home/hww/fusion0820/MF_dataset/Code/rgb_data/case22/';
fileExt = '*.png';
files = dir(fullfile(SamplePaths,fileExt));

len = size(files,1);

for i = 1:len
    fileName = strcat(SamplePaths,files(i).name);
    if contains(fileName,'dg1__enlarge')
        str = strrep(fileName,'/','\');        
        I_ir=im2double(imread(fileName));  
        [I,v1,v2]=rgb2ihs(I_ir); 
        mr_save = '/home/hww/fusion0820/baseline/PMGI_AAAI2020-master/Code_PMGI/Medical/pet-mri/pet_I/';         
        save_path = strcat(mr_save,num2str(str(end-6:end-4)),'.png');
        imwrite(I, save_path);
    elseif contains(fileName,'mr2_')
        mr_save = '/home/hww/fusion0820/baseline/PMGI_AAAI2020-master/Code_PMGI/Medical/pet-mri/mri/';   
        copyfile(fileName,mr_save);

    end  
end 

% 
% for num=1:19
%   I_ir=im2double(imread(strcat('',num2str(num-1),'.png')));  
%   [I,v1,v2]=rgb2ihs(I_ir); 
%   imwrite(I, strcat('',num2str(num),'.bmp'));
% %   imwrite(v1,strcat('',num2str(num),'.bmp'));
% %   imwrite(v2,strcat('',num2str(num),'.bmp'));
%   
% endnum2str,