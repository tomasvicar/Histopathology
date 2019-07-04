import os


patch_path=r'D:\MPI_CBG\data_plice\patches'


file_names=[]
for root, dirs, files in os.walk(patch_path):
        for name in files:
            if name.endswith(".tif"):
                file_names.append(root+'\\'+name)
                
i=0                
for file_name in file_names:
    i=i+1
    print(str(i)+'//'+str(len(file_names)))
    os.remove(file_name)