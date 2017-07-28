import csv
import gdal
import osr
import numpy as np
import pandas as pd

def output_lables_to_tif(z_ps, class_file_path, rows, cols):
    
    # Creates an empty Frame to store classification results
    z_p_frame = pd.DataFrame(data = np.empty((rows*cols,1)), columns = ['lables'])
      
    # Fill the Frame with lables values
    z_p_frame.loc[np.arange(0, cols*rows, 1)] = np.array(z_ps).reshape(len(z_ps),1)


    # Convert the result Frame to Matrix
    result_matrix = np.empty((rows,cols))
    for i in range(0,rows):
        result_matrix[i,:] = z_p_frame.loc[i*cols:(i+1)*cols-1].T 
    
    # Output result in "GeoTiff" format           
    driver=gdal.GetDriverByName("GTiff")
    driver.Register()
    outDataset = driver.Create(class_file_path, cols, rows, 1, gdal.GDT_Float64)
    
    # Define the projection coordinate system
    outDataset.SetGeoTransform( [ 422985, 30, 0, 3439515, 0, -30 ] )
    proj = osr.SpatialReference()
    proj.ImportFromProj4("+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0")
    outDataset.SetProjection( proj.ExportToWkt())
    outDataset.GetRasterBand(1).WriteArray(result_matrix)
    outDataset = None  
    print('Done')


def prepare_data(imagery_path, train_file_path):
    # read image
    dataset = gdal.Open(imagery_path)
    dsmatrix = dataset.ReadAsArray(xoff=0, yoff=0, xsize=dataset.RasterXSize, ysize=dataset.RasterYSize)
    
    # read samples
    original_data_list = []
    csv_reader = csv.reader(open(train_file_path, encoding='utf-8'))
    for row in csv_reader:
        original_data_list.append(row)
    original_data_array = np.array(original_data_list)
    
    # split training data into variables and lables 
    x_s = original_data_array[:,2:8] 
    y_s = original_data_array[:,8]
    
    # unfold array into pandas DataFrame
    rows = dsmatrix.shape[1]
    cols = dsmatrix.shape[2]
    data_array = dsmatrix[:,0,:]
    for irow in range(1,rows):
        tempmatirx = dsmatrix[:,irow,:]
        data_array = np.hstack((data_array,tempmatirx))
    data_frame = pd.DataFrame(data_array.T)
    
    return x_s, y_s, rows, cols, data_frame


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