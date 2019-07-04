import os


#subfolders = [f.path for f in os.scandir(self.folder_path) if f.is_dir() ]


folder_1=r'D:\MPI_CBG\camelyon16\dataset\test\data'
folder_2=r'D:\MPI_CBG\camelyon16\dataset\test\fg'


file_names1=[]
for root, dirs, files in os.walk(folder_1):
        for name in files:
            if name.endswith(".tif"):
                file_names1.append(root+'\\'+name)
                
                
                
file_names2=[]
for root, dirs, files in os.walk(folder_2):
        for name in files:
            if name.endswith(".tif"):
                file_names2.append(root+'\\'+name)
                
                
for name1 in file_names1:
    for name2 in file_names2:
        s=name2.split('\\')
        ss=s[-1][-7:-4]
        if  ss in name1:
            s[-1]=name1.split('\\')[-1]
            newname=''
            for i in range(len(s)):
                newname+= '\\'+s[i]
            newname=newname[1:]          
            os.rename(name2, newname)
    
    
#    
#    
#folder_1=r'D:\MPI_CBG\camelyon16\dataset\test\data'
#folder_2=r'D:\MPI_CBG\camelyon16\patches\test'
#
#
#file_names1=[]
#for root, dirs, files in os.walk(folder_1):
#        for name in files:
#            if name.endswith(".tif"):
#                file_names1.append(root+'\\'+name)
#                
#                
#                
#file_names2 = [f.path for f in os.scandir(folder_2) if f.is_dir() ]
#                
#                
#for name1 in file_names1:
#    for name2 in file_names2:
#        s=name2.split('\\')
#        if s[-1] in name1:
#            s[-1]=name1.split('\\')[-1][:-4]
#            newname=''
#            for i in range(len(s)):
#                newname+= '\\'+s[i]
#            newname=newname[1:]          
#            os.rename(name2, newname)    
    