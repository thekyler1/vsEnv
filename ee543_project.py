import numpy as np
import os
import torch
import torch.nn as nn




path_trainSet = f"//Users//necatiisik//Downloads//lfw_dataList//peopleDevTrain.txt"
path_testSet = f"//Users//necatiisik//Downloads//lfw_dataList//peopleDevTest.txt"

testSet_list = np.genfromtxt(path_testSet,dtype=str, delimiter="\t")
trainSet_list = np.genfromtxt(path_testSet,dtype=str, delimiter="\t")


print(np.shape(my_list))

print(my_list[1,0])

my_string = my_list[1,1]
my_string = my_string.zfill(4)
print(my_string)
print(type(my_string))



