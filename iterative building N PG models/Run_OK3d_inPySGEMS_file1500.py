from contextlib import closing
from copy import deepcopy
from locale import currency
import os
from tkinter import PROJECTING
from matplotlib import lines
import numpy as np
import shutil
import skgstat as skg
import math   
# from pykrige import ok3d
# import skgstat as skg
# from pykrige import OrdinaryKriging3D
from os.path import join as join_path
from xml.dom.expatbuilder import FilterVisibilityController

from pysgems.algo.sgalgo import XML
from pysgems.dis.sgdis import Discretize
from pysgems.io.sgio import PointSet
from pysgems.plot.sgplots import Plots
from pysgems.sgems import sg
if __name__ == '__main__':
    FolderPath = os.path.dirname(os.path.abspath(__file__)).replace('\\','/')+'/'
    cwd = FolderPath
    data_dir = FolderPath+'RandomParticleDisSGEMS/'
    rdir = FolderPath+'resultsLY/'
    all_files = os.listdir(data_dir)
    SamFiles = []
    Nx,Ny,Nz,Xc,Yc,Zc,gsx,gsy,gsz = np.loadtxt(FolderPath+'grid_Info2.txt',dtype=float,delimiter=' ')
    Xmax = Xc+(Nx-0)*gsx
    Ymax = Yc+(Ny-0)*gsy
    Zmax = Zc+(Nz-0)*gsz
    numbp = int(Nx*Ny*Nz)
    Non_Inter_array = np.zeros(numbp)
    for nnni in all_files:
        if os.path.splitext(nnni)[1].lower() == '.txt':
            SamFiles.append(nnni)
    # print(SamFiles)
    if not os.path.exists(rdir):
        os.mkdir(rdir)
    f_log = open(FolderPath+'run_logs.txt','w')
    for filei in SamFiles:
        samp_file_i_path = data_dir+filei
        pjt = sg.Sgems(project_name="test",project_wd=cwd,res_dir=rdir)
        ps = PointSet(project=pjt,pointset_path=samp_file_i_path)
        grid_temp = Discretize(project=pjt,dx=gsx,dy=gsy,dz=gsz,xo=Xc,yo=Yc,zo=Zc,x_lim=Xmax,y_lim=Ymax,z_lim=Zmax)
        xyz_temp = ps.xyz
        pl = Plots(project=pjt)
        # pl.plot_coordinates()
        # print('所有有效列的名字如下行：：：：：')
        # print(pjt.point_set.columns)
        col_name_current = pjt.point_set.columns
        algorithm_dir = FolderPath+'algorithms/'
        al = XML(project=pjt,algo_dir=algorithm_dir)
        al.xml_reader("kriging")
        # al.show_tree()
        for cols_i in range(4,len(col_name_current)):
            pro_cur = ps.raw_data[:,cols_i]
            Var_name = 'spherical'
            if np.max(pro_cur)-0.000001>0:
                VVV = skg.Variogram(xyz_temp,pro_cur)
                Var_name = VVV._model_name
                VVV.model = 'spherical'
                paras_var = VVV.parameters
                paras_var_new = [str(paras_var[0]),str(paras_var[1]),str(paras_var[2])]#按次序依次为变程、总基台值、块金
            else:
                paras_var_new = ['100','1','0']
            cols_i_name = col_name_current[cols_i]
            al.xml_update("Hard_Data","grid",cols_i_name+"_grid")
            al.xml_update("Hard_Data","property",cols_i_name)
            al.xml_update('Variogram//structure_1//ranges', new_attribute_dict={'max': paras_var_new[0], 'medium': paras_var_new[0], 'min': '3'})
            # al.xml_update('Variogram//structure_1',new_attribute_dict={'contribution':'1', 'type':Var_name})
            # al.show_tree()
            ps.export_01(cols_i_name)#输出01后才可以展开插值
            if np.max(pro_cur)-0.000001>0:
                
                pjt.write_command()
                pjt.run()
                if not os.path.exists(rdir+"results.grid"):
                    # os.mkdir(rdir+'/All_InterResult/'+filei.replace('.txt',"_")+cols_i_name+'.result')
                    f_p = open(rdir+'/All_InterResult/'+filei.replace('.txt',"_")+cols_i_name+'.result','w')
                    f_p.write('computation_grid (110x52x100)\n1\nkriging')
                    for tyi in Non_Inter_array:
                        f_p.write(str(tyi)+'\n')
                    f_p.close()
                    f_log.write(filei+'文件的颗粒下：'+cols_i_name+'应该插值，但没有插值，请确认该结果。。\n')
                else:
                    shutil.move(rdir+"results.grid",rdir+'/All_InterResult/'+filei.replace('.txt',"_")+cols_i_name+'.result')
            else:
                # if not os.path.exists(rdir+'/All_InterResult/'+filei.replace('.txt',"_")+cols_i_name+'.result'):
                #     os.mkdir(rdir+'/All_InterResult/'+filei.replace('.txt',"_")+cols_i_name+'.result')
                f_p = open(rdir+'/All_InterResult/'+filei.replace('.txt',"_")+cols_i_name+'.result','w')
                f_p.write('computation_grid (110x52x100)\n1\nkriging')
                for tyi in Non_Inter_array:
                    f_p.write(str(tyi)+'\n')
                f_p.close()
            os.remove(rdir+cols_i_name+'.sgems')
            # os.rename("results.grid",cols_i_name+'.result')
            # shutil.move(rdir+cols_i_name+'.result',rdir+'/All_InterResult/'+cols_i_name+'.result')
            # name_Krig = "results"#filei.replace('.txt','_')+'big200'
            # result_file_Krig = join_path(rdir,f"{name_Krig}.grid")
            # save = "kriging"

            # name_Krig_var = "results_var"#filei.replace('.txt','_')+'big200_var'
            # result_file_Krig = join_path(rdir,f"{name_Krig}.grid")
            # save = "kriging_var"
        print(filei+'运行完成了。。。')





