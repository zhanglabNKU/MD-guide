% This procedure loads a sequence of images
%
% Arguments:
%   'path', refers to a directory which contains a sequence of images
%   'reduce' is an optional parameter that controls downsampling, e.g., reduce = .5
%   downsamples all images by a factor of 2.
%
% tom.mertens@gmail.com, August 2007
%

function I = load_images(path,path2,reduce)

if ~exist('reduce')
    reduce = 1;
end

if (reduce > 1 || reduce <= 0)
    error('reduce must fulfill: 0 < reduce <= 1');
end

% find all JPEG or PPM files in directory
% files = dir([path '/*.tif']);
% N = length(files);
% if (N == 0)
%     files = dir([path '/*.jpg']);
%     N = length(files);
%     if (N == 0)
%     files = dir([path '/*.gif']);
%     N = length(files);
%     if (N == 0)
%     files = dir([path '/*.bmp']);
%     N = length(files);
%     if (N == 0)
%     files = dir([path '/*.png']);
%     N = length(files);
%     if (N == 0)
%     error('no files found');
%     end
%           end
%          end
%     end
% end
N=2;
% allocate memory
sz = size(imread(path));
r = floor(sz(1)*reduce);
c = floor(sz(2)*reduce);
if length(sz)==3
    I=zeros(r,c,3,N);
else
I = zeros(r,c,N);
end
%%Ë÷ÒýÍ¼Ïñ×ª»»Îª»Ò¶ÈÍ¼Ïñ
%info=imfinfo(path)
% [I1,img_ct]  = imread(path);
% [I2,img_mr]  = imread(path2);
% img1=ind2gray(I1,img_ct);
% img2=ind2gray(I2,img_mr);
%%»Ò¶ÈÍ¼Ïñ
% img1 = imread(path);
% img2 = imread(path2);
% 
% I(:,:,1) = img1;
% I(:,:,2) = img2;
% I=uint8(I);
% end

    img1 = imread(path);
    img2 = imread(path2);
    if size(img1,3)==3
    I(:,:,:,1) =img1;   
    I(:,:,:,2) =img2;
    else
    I(:,:,1) = img1; 
    I(:,:,2) = img2;
    end
    I=uint8(I);

end
