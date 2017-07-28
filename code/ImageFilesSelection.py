# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 17:00:36 2016

Purpose:Select files basised on date and coordinate(for GF data)

@author: XJG
"""

import os
import operator
import shutil

#文件对象
class fileObj:
    fileDate = 0
    fileLon = 0.0
    fileLat = 0.0
    fileName = ''
    def __init__(self, fName):
        self.fileName = fName
        self.fileType = fName[0:7]
        self.fileLon = float(fName[10:14])
        self.fileLat = float(fName[17:20])
        self.fileDate = int(fName[22:30])

#读取原文件列表
def getOriginalFileList(dirStr):
    fileObjectsList = []
    for i in os.walk(dirStr):   #递归遍历
        for j in i[2]:
            fileFormatStr = j[-2:]
            if fileFormatStr == 'gz':
                fileObjectsList.append(fileObj(j))
    return fileObjectsList     #包含文件名字符串的列表

#选择符合要求的文件
def selectFiles(fileObjectsList, limitsList):
    selectedFilesList = []
    for rsFile in fileObjectsList:
        if ((limitsList[0] < rsFile.fileLon < limitsList[2])&(limitsList[1] < rsFile.fileLat < limitsList[3])&(limitsList[4] <= int(str(rsFile.fileDate)[4:5]) <= limitsList[5])):
            selectedFilesList.append(rsFile)
    return selectedFilesList

#将文件选择的结果复制到指定目录，同时将文件名输出
def writeResult(dirStr,selectedFilesList):
    #对文件进行排序
    cmpfun = operator.attrgetter('fileDate')
    selectedFilesList.sort(key=cmpfun)
    
    #检查文件是否已经存在
    targetPath=os.path.join(dirStr, 'SelectedFiles')
    if os.path.exists(targetPath):
        return 0
    os.mkdir(targetPath)

    print('共检索到符合要求的文件'+str(len(selectedFilesList))+'个,正在执行拷贝，请稍后……')
    with open(targetPath + '/FileList.txt', 'w') as f:
        for resultObj in selectedFilesList:
            f.write(str(resultObj.fileDate) + '\t' + resultObj.fileName + '\n')
            for i in os.walk(dirStr):
                for j in i[2]:
                    if j==resultObj.fileName:
                        shutil.copy(os.path.join(dirStr,j), os.path.join(targetPath,j))     #文件复制
    return 1

if __name__ == "__main__":
    #在此输入参数，路径和限制条件
    dirStr = r'K:\hljdfh1102'
    LonMin = 128
    LatMin = 47.5
    LonMax = 130.4
    LatMax = 48.5
    MonthMin = 1
    MonthMax = 12
    limitsList = [LonMin, LatMin, LonMax, LatMax, MonthMin, MonthMax]

    fileObjectsList = getOriginalFileList(dirStr)
    selectedFilesList = selectFiles(fileObjectsList, limitsList)
    
    if selectedFilesList:   #判断是否为空，如果为空，则List为false
        resultState=writeResult(dirStr,selectedFilesList)
        if resultState==0:
            print('请先删除SelectedFiles文件夹！')
    else:
        print('没有符合要求的文件！')
        
    print('完毕！')