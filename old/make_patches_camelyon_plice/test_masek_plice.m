clc;clear all;close all

listing={};

s='D:\MPI_CBG\data_plice\dataset\train\data';
listing_tmp=dir(s);
listing_tmp2={};
for k=3:length(listing_tmp)
    listing_tmp2 = [listing_tmp2 [listing_tmp(k).folder  '\' listing_tmp(k).name]];

end
listing=[listing listing_tmp2];

s='D:\MPI_CBG\data_plice\dataset\test\data';
listing_tmp=dir(s);
listing_tmp2={};
for k=3:length(listing_tmp)
    listing_tmp2 = [listing_tmp2 [listing_tmp(k).folder  '\' listing_tmp(k).name]];

end
listing=[listing listing_tmp2];

s='D:\MPI_CBG\data_plice\dataset\valid\data';
listing_tmp=dir(s);
listing_tmp2={};
for k=3:length(listing_tmp)
    listing_tmp2 = [listing_tmp2 [listing_tmp(k).folder '\' listing_tmp(k).name]];

end
listing=[listing listing_tmp2];

listing_tmp={};
for k=1:length(listing)
    if contains(listing{k},'tumor')
        listing_tmp=[listing_tmp listing{k}];
    end
end
listing=listing_tmp;


for name=listing

    name=name{1};
    if strcmp(name,'D:\MPI_CBG\data_plice\dataset\test\data\2017_11_30__0034-4.czi.labeling_tumor.tif')==0
        continue 
    end
   
    
    
    name_mask=name;
    name_mask=strrep(name_mask,'\data\','\mask\');
    name_mask=[name_mask(1:end-4) '_mask.tif'];
    
%     data=imread(name,1,'PixelRegion',{[2200*4 2200*4+300],[606*4 606*4+300]});
%     mask=rgb2gray(imread(name_mask,1,'PixelRegion',{[2200*4 2200*4+300],[606*4 606*4+300]}));
    
    
    
    
    
    data=imread(name,2);
    mask=rgb2gray(imread(name_mask,2));
    
%     mask=imresize(mask,[4383,9442]);
    
    figure()
    hold off
    imshow(data,[])
    hold on
    visboundaries(mask>0)
    title(name)
    
%     imwrite(data,'test_patch.tif')
%     
%     imwrite(uint8((mask>0)*255),'gt.tif')

end



% data=imread("D:\MPI_CBG\camelyon16\dataset\train\data\tumor_104.tif",4);
% mask=imread("D:\MPI_CBG\camelyon16\dataset\train\mask\tumor_104_mask.tif",4);
% 
% x=9770;
% y=2990;
% w=1000;
% 
% imshow(data,[])
% hold on
% visboundaries(mask>0)
% rectangle('Position',[x,y,w,w])
% 
% print('camelyon16_data','-dpng')
% 
% clear data;
% clear mask;
% close all;
% 
% x=x*8;
% y=y*8;
% w=w*8;
% 
% data=imread("D:\MPI_CBG\camelyon16\dataset\train\data\tumor_104.tif",1,'PixelRegion',{[y y+w],[x x+w]});
% mask=imread("D:\MPI_CBG\camelyon16\dataset\train\mask\tumor_104_mask.tif",1,'PixelRegion',{[y y+w],[x x+w]});
% 
% 
% imshow(data,[])
% hold on
% visboundaries(mask>0)
% print('camelyon16_data_zoom','-dpng')




% 
% 
% 
% clc;clear all;close all
% 
% data=imread("D:\MPI_CBG\data_plice\dataset\train\data\2017_11_30__0044-2.czi.labeling_tumor.tif",4);
% mask=imread("D:\MPI_CBG\data_plice\dataset\train\mask\2017_11_30__0044-2.czi.labeling_tumor_mask.tif",4);
% 
% x=659;
% y=1316;
% w=200;
% 
% imshow(data,[])
% hold on
% visboundaries(rgb2gray(mask)>0)
% rectangle('Position',[x,y,w,w])
% 
% print('lungs_data','-dpng')
% 
% clear data;
% clear mask;
% close all;
% 
% x=x*8;
% y=y*8;
% w=w*8;
% 
% data=imread("D:\MPI_CBG\data_plice\dataset\train\data\2017_11_30__0045-1.czi.labeling_tumor.tif",1,'PixelRegion',{[y y+w],[x x+w]});
% mask=imread("D:\MPI_CBG\data_plice\dataset\train\mask\2017_11_30__0045-1.czi.labeling_tumor_mask.tif",1,'PixelRegion',{[y y+w],[x x+w]});
% 
% 
% imshow(data,[])
% hold on
% visboundaries(rgb2gray(mask)>0)
% print('lungs_data_zoom','-dpng')