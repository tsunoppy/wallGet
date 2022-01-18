# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 10:38:49 2022

@author: R18102
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ptick

import os

class rout_1:

    def __init__(self,inpFile,ax,screen):
        self.inpFile=inpFile
        self.ax = ax
        self.screen = screen

    def viewModel(self,node,t,gx,gy,ax):

        #fig = plt.figure(tight_layout=True)
        #ax = plt.axes()         

        x = []
        y = []

        for i in range(0,len(node)):
            x.append(node[i,0])
            y.append(node[i,1])

        for i in range(0,len(node)-1):

            l = (x[i+1]-x[i])**2 + (y[i+1]-y[i])**2
            l = math.sqrt(l)

            cs = (x[i+1]-x[i])/l
            sn = (y[i+1]-y[i])/l
            #th = math.acos(cs)*360.0/2.0/math.pi
            th = math.atan2(sn,cs) * 360.0/2.0/math.pi
            w = l
            h = t
            #print("?",w,h,th)
            fib = patches.Rectangle(xy=(x[i], y[i]-t/2.0), width=w, height=h, \
                                    angle=th, linewidth="0.5", ec='#000000', color="gray", alpha=0.5 )
            ax.add_patch(fib)

        # Gravity Center
        #sg = r_model
        #ax.scatter(self.xg,self.yg, s=sg, color="blue", marker="D")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize="8")
        ax.set_aspect('equal', 'datalim')
        ax.scatter(gx,gy,color='b')
        ax.tick_params(labelsize="8")
        ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        ax.xaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        ax.xaxis.offsetText.set_fontsize(8)
        ax.ticklabel_format(style="sci", axis="x",scilimits=(3,3))
        ax.yaxis.offsetText.set_fontsize(8)
        ax.ticklabel_format(style="sci", axis="y",scilimits=(3,3))
        #ax.set_aspect('equal')
        #plt.plot(0.0,0.0)
        #plt.show()
        #plt.close()


    ########################################################################
    # solve
    def solve(self):

        self.read_data()
        num2 = 0

        #fig = plt.figure(tight_layout=True,dpi=300)
        #ax = plt.axes()
        ax = self.ax

        dx = 0.0
        dy = 0.0
        dxpy = 0.0
        dypx = 0.0

        out1 = []
        out2 = []
        out3 = []
        out4 = []


        for i in range(0,len(self.name)):

            num1 = num2
            num2 = num1 + self.numdata[i]
            #print(num1,num2)
            awx,awy,aw, kx, ky, px, py = \
                self.cal(self.node[num1:num2],self.thn[i])

            dx = dx + awx
            dy = dy + awy
            dxpy = dxpy + awx*py
            dypx = dypx + awy*px

            #print(self.cal(self.node[num1:num2],self.thn[i]))
            #print(\
            #      awx,awy,aw, kx, ky ,px, py  \
            #      )
            #print("{:5s}".format(self.name[i]),"t={:7.0f}".format(self.thn[i]),"awx={:10.2e}".format(awx),"awy={:10.2e}".format(awy),\
            #      "px={:10.0f}".format(px),"py={:10.0f}".format(py))
            out1.append(awx)
            out2.append(awy)
            out3.append(px)
            out4.append(py)
            self.viewModel(self.node[num1:num2],self.thn[i],px,py,ax)

            ax.text(px,py,self.name[i],fontsize=8)
        ppx = dypx/dy
        ppy = dxpy/dx


        num = len(self.name)
        df_out = pd.DataFrame(np.arange(num*6).reshape(num,6),
                          columns=['name','t','awx','awy','px','py'])
        df_out.loc[:,'name']    = self.name
        df_out.loc[:,'t']       = self.thn
        df_out.loc[:,'awx']     = out1
        df_out.loc[:,'awy']     = out2
        df_out.loc[:,'px']      = out3
        df_out.loc[:,'py']      = out4
        out_dir = os.path.dirname(self.inpFile)
        df_out.to_csv(out_dir+"/outdata.csv",header=True,index=None)
        print(df_out)

        print("Total.","Awx={:10.2e}".format(dx),\
              "Awy={:10.2e}".format(dy),\
                  "px={:10.2e}".format(ppx),\
                      "py={:10.2e}".format(ppy)\
                          )


        ax.scatter(ppx,ppy,color='r',marker='D')

        #plt.show()
        #plt.close()
        self.screen.draw()
        #self.viewModel(self.node[num1:num2],self.thn[i])
        return dx,dy,ppx,ppy

    def cal(self,node,t):
        # calculate aw, awx, awy
        #print("hello")
        
        x = []
        y = []
        
        for i in range(0,len(node)):
            x.append(node[i,0])
            y.append(node[i,1])
            
        #print(x,y)
        
        awx = 0.0
        awy = 0.0
        aw = 0.0
        
        kx = 0.0
        ky = 0.0
        
        kxgy = 0.0
        kygx = 0.0
        
        ggx = 0.0
        ggy = 0.0
        
        for i in range(0,len(node)-1):
            
            l = (x[i+1]-x[i])**2 + (y[i+1]-y[i])**2
            l = math.sqrt(l)
            
            #print(l)
            cs = (x[i+1]-x[i])/l
            sn = (y[i+1]-y[i])/l
            
            cs2 = cs**2
            sn2 = sn**2
            
            aw = aw + l * t
            awx = awx + l * cs2 * t
            awy = awy + l * sn2 * t
            
            gx = (x[i] + x[i+1])/2
            gy = (y[i] + y[i+1])/2
            
            ggx = ggx + gx * l * t
            ggy = ggy + gy * l * t
            
            kxgy = kxgy + cs2*t * gy
            kygx = kygx + sn2*t * gx
            
            kx = kx + cs2 * t
            ky = ky + sn2 * t
        
        gx = ggx / aw
        gy = ggy / aw
        
        if ky == 0 :
            px = gx
        else:
            px = kygx/ky
            
        if kx == 0:
            py = gy
        else:
            py = kxgy/kx
        
        return awx,awy,aw, kx, ky, px, py
        
    ########################################################################
    # read data
    def read_data(self):
    
        cntl = self.inpFile
        #"sample.csv"
        df = pd.read_csv(cntl)
        num = len(df)
        print(df)
        
        name=[]
        thn = []
        numdata =[]
        kk = 0
        
        
        for i in range(0,num):
            # read parameter
            if(df.iloc[i,0] == "CNTL"):
                name.append( df.iloc[i,1] )
                thn.append( df.iloc[i,2] )
                node = []
                x =[]
                y =[]
            if(df.iloc[i,0] == "NODE"):
                node.append( (float(df.iloc[i,1]), float(df.iloc[i,2]) ) )
                x.append( (float(df.iloc[i,1]) ) )
                y.append( (float(df.iloc[i,2]) ) )
                #dia.append( str(data[6]).replace(' ','') )
                    
            if(df.iloc[i,0] == "END"):
                numdata.append(len(node))
                if kk == 0:    
                    allnode = np.array(node)
                    #allx = np.array(x)
                else:
                    allnode = np.vstack( (allnode, np.array(node)))
                    #allx = np.vstack( (allx, np.array(x))  )
                kk = kk + 1             
        print("Finish to read!")
        
        print(name,thn)
        print(numdata)
        #print(allnode)
        #print(allnode[numdata[0]:numdata[1]])

        self.name = name
        self.thn = thn
        self.numdata=numdata
        self.node=allnode
        #print(node[1][0])

"""        
obj=rout_1()
obj.read_data()
obj.solve()
"""        

