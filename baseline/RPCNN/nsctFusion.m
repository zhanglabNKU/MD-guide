function imF = nsctFusion(imA, imB, filter_size, Para, NSCTPara)

% NSCT decomposition
coefA = nsctdec( double(imA)/255, NSCTPara.levels, NSCTPara.dfilt, NSCTPara.pfilt);
coefB = nsctdec( double(imB)/255, NSCTPara.levels, NSCTPara.dfilt, NSCTPara.pfilt);
coefF = coefA;
% Fusing approximate (low pass) subband
disp('Fusing LF Subband: Start');
coefF{1,1}= PCNN_Normal(coefA{1,1}, coefB{1,1}, filter_size, Para);
disp('Fusing LF Subband: End');

% Fusing detail (high pass) subband
disp('Fusing HF Subbands: Start');
coefF = fuse_hfs(coefA, coefB, Para, filter_size, coefF, NSCTPara.levels);
disp('Fusing HF Subband: End');

imF = nsctrec(coefF,NSCTPara.dfilt, NSCTPara.pfilt) ;
imF=double(real(double(imF)));
imF=floor(255*imF);
imF=uint8(imF);


