import csv
import gdal
import osr
import numpy as np

def output_lables_to_tif(frame_file, file_path, rows, cols):

    #将结果Frame转成Matrix
    result_matrix = np.empty((rows,cols))
    for i in range(0,rows):
        result_matrix[i,:] = frame_file.loc[i*cols:(i+1)*cols-1].T 
    
    #将结果输出为tiff文件            
    driver=gdal.GetDriverByName("GTiff")
    driver.Register()
    outDataset = driver.Create(file_path,cols,rows,1, gdal.GDT_Float64)
    
    #定义空间参考坐标系
    outDataset.SetGeoTransform( [ 422985, 30, 0, 3439515, 0, -30 ] )
    proj = osr.SpatialReference()
    proj.ImportFromProj4("+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
    outDataset.SetProjection( proj.ExportToWkt())
    outDataset.GetRasterBand(1).WriteArray(result_matrix)
    outDataset = None  
    print('输出完毕')


def read_NaN(NaN_file_path):
    del_list = []
    csv_reader = csv.reader(open(NaN_file_path, encoding='utf-8'))
    for row in csv_reader:
        del_index = (int(row[0])-1) * 1901 + int(row[1]) - 1
        del_list.append(del_index) 
    
    return del_list

def features_str_to_float(str_array):
    float_array=np.empty(str_array.shape)
    for i in range(0,str_array.shape[0]):
        for j in range(0,str_array.shape[1]):
            float_array[i,j]=float(str_array[i,j])
    
    return float_array

def lables_str_to_int(str_array):
    int_array=np.zeros(str_array.shape)
    for i in range(0,str_array.shape[0]):
        int_array[i]=int(str_array[i])
    
    return int_array