clc;clear all;close all;

data_path='C:\Users\Tom\Desktop\deep_drazdany\histolky\both_datasets\dataset\test_plice\data\';
result_path='C:\Users\Tom\Desktop\deep_drazdany\histolky\both_datasets\results\1s_no_pixel_aug2\';
lvl=4;

listing=dir([data_path '*.tif']);
listing={listing(:).name};

listing=listing(5);

for l=listing
    data_name=[data_path l{1}];
    mask_name= strrep(data_name,'\data\','\mask\');
    mask_name=[mask_name(1:end-4) '_mask.tif'];
    result_name=[result_path l{1}];
    
    data=imread(data_name,lvl);
    if contains(mask_name,'tumor')
        mask=imread(mask_name,lvl);
        mask=mask(:,:,1)>0;
    else
        mask=zeros(size(data,1),size(data,2));
    end
    result=imread(result_name,lvl);
    
    imshow(result,[])
    
end
