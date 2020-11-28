close all;
clear all;
clc;
 pet_path = 'F:\pet合成\论文2\20200714\datasets-master\MFI-WHU-master\source2\';
 mr_path = 'F:\pet合成\论文2\20200714\datasets-master\MFI-WHU-master\source1\';
%  pet_path = 'F:\pet合成\论文2\20200714\datasets-master\LytroDataset\';
%  mr_path = 'F:\pet合成\论文2\20200714\datasets-master\LytroDataset\NaturalMultifocusImages\';
 fileExt = '*.jpg';
 files = dir(fullfile(pet_path,fileExt)); 
 files_mr =  dir(fullfile(mr_path,fileExt)); 
 len1 = size(files_mr,1);
 
 
 for k=1:len1;
     if contains(files_mr(k).name,'jpg')
         A  = imread(strcat(pet_path,files_mr(k).name));
         B  = imread(strcat(mr_path,files_mr(k).name));
%          imwrite(B,strcat('results\',files(k).name(1:end-4),'.png'));
        if size(A)~=size(B)
            error('two images are not the same size.');
        end
        
        model_name = 'model/cnnmodel.mat';

        F=CNN_Fusion(A,B,model_name);

%         figure,imshow(F);
        imwrite(F,strcat('F:/pet合成/论文2/20200714/CNN-Fusion-master/results/',files_mr(k).name(1:end-4),'.png'));
     end
end
 

