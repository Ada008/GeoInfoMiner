import csv
import gdal
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


from imp import reload
import sys
sys.path.append(r'E:\MyProgram\Python\ing\ing')
import Xutils
reload(Xutils)

train_file_path = r'E:\Research\Experiment\qianshan\features\Indiv\samples\Samples_cord.csv'
imagery_path = r'E:\Research\Experiment\qianshan\features\Indiv\nor_l8121.tif'
class_save_path = r'E:\Research\Experiment\qianshan\features\Indiv\IndivKNeighbors_class.tif'

n_neighbors=15

#读取影像
dataset = gdal.Open(imagery_path)
dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)

#读取样本
original_data_list = []
csv_reader = csv.reader(open(train_file_path, encoding='utf-8'))
for row in csv_reader:
    original_data_list.append(row)
original_data_array = np.array(original_data_list)
x_s = original_data_array[:,2:8] 
y_s = original_data_array[:,8]

#影像数据展开
rows = dsmatrix.shape[1]
cols = dsmatrix.shape[2]
pars = dsmatrix.shape[0]
data_array = dsmatrix[:,0,:]
for irow in range(1,rows):
    tempmatirx = dsmatrix[:,irow,:]
    data_array = np.hstack((data_array,tempmatirx))
data_frame = pd.DataFrame(data_array.T)

#训练分类器
clf_KNeighbors = KNeighborsClassifier(n_neighbors = n_neighbors, weights='distance')
clf_KNeighbors.fit(x_s, y_s)
#用分类器进行预测
z_ps=clf_KNeighbors.predict(data_frame)

#创建一个零值的Frame用来存储分类结果
z_p_frame = pd.DataFrame(data = np.empty((rows*cols,1)), columns = ['lables'])
  
#z_p原是1维数组，将其转为2维后赋值
z_p_frame.loc[data_frame.index] = np.array(z_ps).reshape(len(z_ps),1)

#输出分类结果
Xutils.output_lables_to_tif(z_p_frame,class_save_path,rows,cols)