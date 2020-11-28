clc,clear
%%%% gray image fusion
% I = load_images( '.\sourceimages\grayset',1); 
% F = GFF(I);
% imshow(F);
%%%% color image fusion
% I = load_images( '.\Sourceimages\colourset',1);

% path = 'F:\pet合成\论文2\20200714\IFCNN-master\Code\datasets\MDDataset\';
% filename = 'F:\pet合成\论文2\20200714\IFCNN-master\Code\datasets\MDDataset\test_list.txt';
% [im1,im2]=textread(filename,'%s%s');

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
        str=strcat(pet_path,char(files(k).name));
        str2=strcat(mr_path,char(files(k).name));
%         str2=strcat(mr_path,strrep(files(k).name,'_1','_2'));
        I = load_images( char(str),char(str2),1); 
        F = GFF(I);
        fusion_name=strcat('F:\文章1\Medical-Image-Fusion-master\baseline\baseline\GFF_1.0\result\multi-focus\',files(k).name(1:end-4),'.png'); 
        imwrite(F,char(fusion_name));
         end
end
 
 
% for k=1:size(im1);
%     str=strcat(path,char(im1(k)));
%     str2=strcat(path,char(im2(k)));
%     I = load_images( char(str),char(str2),1); 
%     F = GFF(I);
%     fusion_name=strrep(im1(k),'_1','_result'); 
%     fusion_name=strrep(fusion_name,'c','./result/c'); 
% 
%     imwrite(F,char(fusion_name));
% %     str=strrep(im1(k),'./data','F:\Medical-Image-Fusion-master\data');
% %     str=strrep(str,'/','\');              
% %     str2=strrep(str,'ct1','mr2');
% 
% %     fusion_name=strrep(im1(k),'gif','tif'); 
% %     fusion_name=strrep(fusion_name,'/','_'); 
% %     fusion_name=strrep(fusion_name,'._data','./results/');
% %     imwrite(F,char(fusion_name));
% %     figure,imshow(F);
% end