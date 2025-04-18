from email import header
from random import seed
import time
from copy import deepcopy
import os
import numpy as np
import math
from scipy.interpolate import Rbf
from os.path import join as join_path
import json

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    Cur_path = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+'/'
    Parent_path = Cur_path[:Cur_path.rfind('/',0,-2)]+'/'
    # print(Parent_path)
    gs_rand_res_path = Parent_path+'1_MergedAll_particleDis/'
    YX_result_path = Parent_path+'merged_for_YX/'
    w_e_path = Parent_path+'2_w_e_model/'
    train_data_name = 'a0.1-0.2_validData.txt'###########################
    train_data = np.loadtxt(Cur_path+train_data_name,dtype=float,delimiter='\t',skiprows=1)
    xx = train_data[:,:-1]
    yy = train_data[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    xx_scale = scaler.fit_transform(xx)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # optimal_paras = {}
    with open(Cur_path+'a1-2_FittingParameters.json','r') as file:#################################
        optimal_paras = json.load(file)
    pre_rdir = Cur_path+'predicted_result/'
    if not os.path.exists(pre_rdir):
        os.mkdir(pre_rdir)
    file_did_all = os.listdir(pre_rdir)
    if isinstance(optimal_paras['hidden_layer_sizes'],int) == True:
        fit_model = MLPRegressor(hidden_layer_sizes=(optimal_paras['hidden_layer_sizes'],),activation=optimal_paras['activation'],
                            solver=optimal_paras['solver'],alpha=optimal_paras['alpha'],batch_size=optimal_paras['batch_size'],max_iter=25000)
    else:
        fit_model = MLPRegressor(hidden_layer_sizes=(optimal_paras['hidden_layer_sizes']),activation=optimal_paras['activation'],
                            solver=optimal_paras['solver'],alpha=optimal_paras['alpha'],batch_size=optimal_paras['batch_size'],max_iter=25000)
    print('模型拟合已完成，进入到预测阶段。。。。')
    # fit_model.fit(X_train_scaled,y_train)
    fit_model.fit(xx_scale,yy)
    pred1_train = fit_model.predict(X_train_scaled)
    mse1 = mean_squared_error(pred1_train,y_train)
    f_out = open(Cur_path+'Mean_square_error_of_trainedData.txt','w')
    f_out.write('mean square error between the predicted result and observated result of train part::::::'+str(mse1)+'\n\n\n\n')
    pred1_test = fit_model.predict(X_test_scaled)
    mse2 = mean_squared_error(pred1_test,y_test)
    f_out.write('mean square error between the predicted result and observated result of test part::::::'+str(mse2)+'\n\n\n\n')
    weights = fit_model.coefs_
    f_in1 = open(Cur_path+train_data_name,'r')
    all_cols = f_in1.readline()
#     Manual_test = np.array([[0.013230018,0.001445682,0.312667924,84.45041642,15.22223996,0.638526618],
# [0.013996502,0.001212734,0.426150926,84.7140051,14.84463473,0.628719032],
# [0.014116741,0.000610919,0.624758465,85.20213512,14.15837875,0.621693134],
# [0.011214533,0.001841287,0.735104261,86.62330764,12.62853227,0.623777866],
# [0.009960043,0.001628443,0.733268358,86.30366546,12.9514777,0.626686156],
# [0.014285272,0.012218761,0.8421041,85.50062638,13.63076549,0.630667627],
# [0.014444393,0.020839253,0.955286862,84.67871016,14.33071933,0.636467278]])
#     Manual_test=scaler.transform(Manual_test)
#     tttt= fit_model.predict(Manual_test)
    all_cols = all_cols.replace(',',' ').replace('\t',' ').replace('  ',' ')
    all_col_name = all_cols.split(' ')
    Predicted_YX_valid = [1,2,3]
    for i, layer_weights in enumerate(weights):
        f_out.write(f"Hidden layer {i} weights:\n")
        # f_out.write(layer_weights)
        for laywi in layer_weights:
            if i == 0:
                laywi_index = layer_weights.tolist().index(laywi.tolist())
                f_out.write(all_col_name[laywi_index]+':\t')
            f_out.write('\t'.join(laywi.astype(str).tolist())+'\n')
        f_out.write('\n\n')
    
    f_in1.close()
    f_out.close()
    #如果首行中有‘孔隙比'名称，则读取其模型，不存在则不读取，，，，相应的’含水量'模型也是如此。。。。。
    e_ested_res = np.array([])
    w_ested_res = np.array([])
    if all_cols.find(' e ') > -1 or all_cols.find('\te\t') > -1 or all_cols.find('孔隙比') > -1 or all_cols.find('空隙比') > -1:
        e_ested_res = np.loadtxt(w_e_path+'e_model_OK_line.result',dtype=float,skiprows=3)
        # if e_ested_res.shape[1] == 1:
        #     e_ested_res = e_ested_res.astype(float)
        # else:
        #     e_ested_res = e_ested_res[:,0].astype(float)
    if all_cols.find(' w ') > 0 or all_cols.find('含水量') > 0 or all_cols.find('含水率') > 0:
        w_ested_res = np.loadtxt(w_e_path+'w_model_OK_line.result',dtype=float,delimiter=' ',skiprows=3)
        # if w_ested_res.shape[1] == 1:
        #     w_ested_res = w_ested_res.astype(float)
        # else:
        #     w_ested_res = w_ested_res[:,0].astype(float)
    

    fir_cols_name = all_cols.replace('>','b').split(' ')[0]
    gsDis_all_file = []
    temp_all = os.listdir(gs_rand_res_path)
    for nnni in temp_all:
        if os.path.splitext(nnni)[1].lower() == '.result':
            gsDis_all_file.append(nnni)
    del temp_all
    ParticleAll = ['b200', 'b20', 'b2', 'b0.5', 'b0.25', 'b0.075', 'b0.005', 'l0.005']
    fir_cols_name_sub_index = ParticleAll.index(fir_cols_name)
    gsDis_count = len(ParticleAll)
    file_did_all = []
    for randfilei in gsDis_all_file:
        if randfilei.replace('.result','_predicted.result') not in file_did_all:
            print('开始处理的文件为：：：：：'+randfilei)
            temp_result = np.loadtxt(gs_rand_res_path+randfilei,dtype=float,skiprows=gsDis_count+2)
            #YX_out = np.loadtxt(YX_result_path+randfilei,dtype=int,delimiter=' ',skiprows=3)
            #YX_out_bool = np.zeros(YX_out.shape[0])
            #for yxI in Predicted_YX_valid:
             #   YX_out_bool = np.where(YX_out==int(yxI),1,YX_out_bool)
             #        if temp_result.shape[1] == gsDis_count:
             #            temp_result = temp_result.astype(float)
             #        else:
             #            temp_result = temp_result[:,:-1].astype(float)
            temp_result = np.where(temp_result<0,0,temp_result)
            temp_result = temp_result / np.sum(temp_result,axis=1,keepdims=True) * 100
            temp_result[:,fir_cols_name_sub_index] = np.sum(temp_result[:,:fir_cols_name_sub_index+1],axis=1)
            sample_gsDis = temp_result[:,fir_cols_name_sub_index:]
            if w_ested_res.size>0:
                sample_gsDis = np.c_[sample_gsDis,w_ested_res]
            if e_ested_res.size>0:
                sample_gsDis = np.c_[sample_gsDis,e_ested_res]
            sample_gsDis_scaled = scaler.transform(sample_gsDis)
            # test_150770 = sample_gsDis_scaled[150770:150800,:]
            pred_out_cur = fit_model.predict(sample_gsDis_scaled)#######X_train_scaled = scaler.fit_transform(X_train)

            # ee = fit_model.predict(test_150770)
            # pred_out_cur = np.where(YX_out_bool==1,pred_out_cur,-9999)
            np.savetxt(Cur_path+'predicted_result/'+randfilei.replace('.result','_predicted.result'),pred_out_cur,fmt='%.5f',delimiter=' ',newline='\n',header=randfilei+'\n1\n'+train_data_name[:-4])
            print(randfilei+'运行完成了。。。。。。')
    print('工程运行完成了。。。。。')

    #os.system('shutdown -s -t 180')
