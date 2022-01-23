# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ptick

import os

class rout_1:

    def __init__(self,inpFile,ax,screen,ax2,screen2,ax3,screen3):

        # input file
        self.inpFile=inpFile

        # for model plot tag
        self.ax = ax
        self.screen = screen

        # for ax-ay plot tag
        self.ax2 = ax2
        self.screen2 = screen2

        # for rex-rey plot tag
        self.ax3 = ax3
        self.screen3 = screen3

        #
        self.read_data()

    ########################################################################
    def grav(self):
        return self.xg, self.yg, self.reqA

    ########################################################################
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

        #plt.plot(0.0,0.0)
        #plt.show()
        #plt.close()

    ########################################################################
    # calculation aw at any angle
    #def ecc(self):



    ########################################################################
    # calculation aw at any angle
    def axay(self):

        ndiv = 32
        delth   = 2.0 * math.pi / ndiv

        x = []
        y = []
        out_alpha = []

        rex_ratio = []
        rey_ratio = []

        for i in range(0,ndiv+1):

            th = delth * i
            out_alpha.append(th)
            dx,dy,ppx,ppy,dr,ex,ey = self.solve2(th)

            x.append(dx * math.cos(th))
            y.append(dx * math.sin(th))

            rex_ratio.append( ex * math.cos(th) )
            rey_ratio.append( ex * math.sin(th) )

        aw = np.sqrt( np.array(x)**2 + np.array(y)**2 )


        #fig = plt.figure(tight_layout=True)
        #ax = plt.axes()
        ax = self.ax2

        if self.reqA == -99:
            draw_circle = plt.Circle((0.0, 0.0), np.min(aw),fill=False)
        else:
            draw_circle = plt.Circle((0.0, 0.0), float(self.reqA), fill=False)

        ax.add_artist(draw_circle)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_aspect('equal', 'datalim')
        ax.set_xlabel('awx, mm2',fontsize="8")
        ax.set_ylabel('awy, mm2',fontsize="8")
        ax.axhline(y=0,color='black',linewidth=0.5,linestyle='--')
        ax.axvline(x=0,color='black',linewidth=0.5,linestyle='--')

        ax.ticklabel_format(style="sci", axis="both",scilimits=(0,0))
        ax.xaxis.offsetText.set_fontsize(8)
        ax.yaxis.offsetText.set_fontsize(8)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize="8")

        #ax.set_xlim(0,)
        #ax.set_ylim(0,)

        #plt.plot(x,y)
        ax.plot(x,y,label='Wall Area')
        ax.legend(loc="upper right",fontsize="6")
        #plt.show()
        self.screen2.draw()

        num = len(x)
        df_out = pd.DataFrame(np.arange(num*3).reshape(num,3),
                          columns=['th','aw_costh','aw_sinth'])
        df_out.loc[:,'th']           = out_alpha
        df_out.loc[:,'aw_costh']     = x
        df_out.loc[:,'aw_sinth']     = y
        out_dir = os.path.dirname(self.inpFile)
        df_out.to_csv(out_dir+"/awx-awy_out.csv",header=True,index=None)


        ########################################################################

        self.ecc(rex_ratio,rey_ratio)

    def ecc(self,rex_ratio,rey_ratio):

        ax3 = self.ax3
        #ax3.clear()

        draw_circle = []
        draw_circle.append( plt.Circle((0.0, 0.0), 0.05, fill=False) )
        draw_circle.append( plt.Circle((0.0, 0.0), 0.1, fill=False) )
        draw_circle.append( plt.Circle((0.0, 0.0), 0.15, fill=False) )

        for i in range(0,len(draw_circle)):
            ax3.add_artist(draw_circle[i])

        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.set_aspect('equal', 'datalim')
        ax3.set_xlabel('Rex',fontsize="8")
        ax3.set_ylabel('Rey',fontsize="8")
        ax3.axhline(y=0,color='black',linewidth=0.5,linestyle='--')
        ax3.axvline(x=0,color='black',linewidth=0.5,linestyle='--')

        ax3.legend(fontsize=8)
        ax3.tick_params(labelsize="8")
        ax3.set_xlim(-0.2,0.2)
        ax3.set_ylim(-0.2,0.2)

        #plt.plot(x,y)
        ax3.plot(rex_ratio,rey_ratio,label='Eccentricity Ratio')
        ax3.legend(loc="upper right",fontsize="6")
        #plt.show()
        self.screen3.draw()


    ########################################################################
    def model(self):

        #self.read_data()
        ax = self.ax

        num2 = 0

        for i in range(0,len(self.name)):

            num1 = num2
            num2 = num1 + self.numdata[i]
            #print(num1,num2)
            awx,awy,aw, kx, ky, px, py , \
                aw_e,awx_e,awy_e,gx_e,gy_e = \
                self.cal(self.node[num1:num2],self.thn[i])
            self.viewModel(self.node[num1:num2],self.thn[i],px,py,ax)
            ax.text(px,py,self.name[i],fontsize=8)


        if self.xg != -99:
            self.xg = float(self.xg)
            self.yg = float(self.yg)
            ax.scatter( self.xg, self.yg, color='r', marker=',')
            ax.text(self.xg,self.yg," G.C",fontsize=8)

        dx,dy,ppx,ppy,dr,ex,ey = self.solve2(0.0)
        ax.scatter(ppx,ppy,color='r',marker='D')
        #plt.show()
        #plt.close()

        self.screen.draw()

    ########################################################################
    def rotate(self,x,y,th):

        xb =  math.cos(th) * x + math.sin(th) * y
        yb = -math.sin(th) * x + math.cos(th) * y

        return xb,yb

    ########################################################################
    def solve2(self,alpha):

        #self.read_data()
        num2 = 0

        nodeb = []
        for i in range(0,len(self.node)):
            nodeb.append( self.rotate(self.node[i][0],self.node[i][1],alpha) )
        xg,yg = self.rotate(self.xg,self.yg,alpha)

        #print(nodeb)
        nodeb = np.array(nodeb)

        # 壁量、剛心位置の計算
        for i in range(0,len(self.name)):

            num1 = num2
            num2 = num1 + self.numdata[i]

            tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7,\
                aw_e,awx_e,awy_e,gx_e,gy_e = \
                self.cal(nodeb[num1:num2],self.thn[i])

            if i ==0 :
                aw  = aw_e
                awx = awx_e
                awy = awy_e
                gx  = gx_e
                gy  = gy_e
            else:
                aw = np.hstack([aw,aw_e])
                awx = np.hstack([awx,awx_e])
                awy = np.hstack([awy,awy_e])
                gx = np.hstack([gx,gx_e])
                gy = np.hstack([gy,gy_e])

        px = np.dot(awy,gx) / np.sum(awy)
        py = np.dot(awx,gy) / np.sum(awx)

        if self.xg != -99:
            ex = abs(xg-px)
            ey = abs(yg-py)
        else:
            ex = 0.0
            ey = 0.0

        xb = gx - px
        yb = gy - py

        xb2 = np.square(xb)
        yb2 = np.square(yb)

        kR = np.dot(awx,yb2) + np.dot(awy,xb2)

        rex = math.sqrt( kR / np.sum(awx) )
        rey = math.sqrt( kR / np.sum(awy) )

        rrex = ey/rex
        rrey = ex/rey

        # data making
        #num = len(self.name)
        num = len(awx)
        df_out = pd.DataFrame(np.arange(num*5).reshape(num,5),
                          columns=['aw','awx','awy','gx','gy'])
        df_out.loc[:,'aw']      = aw
        df_out.loc[:,'awx']     = awx
        df_out.loc[:,'awy']     = awy
        df_out.loc[:,'gx']      = gx
        df_out.loc[:,'gy']      = gy
        out_dir = os.path.dirname(self.inpFile)
        df_out.to_csv(out_dir+"/outdata.csv",header=True,index=None)

        #print("solve2",np.sum(aw),np.sum(awx),np.sum(awy))
        #print("px,py",px,py)
        #print("Rex,Rey",rrex,rrey)

        return np.sum(awx),np.sum(awy),px,py,kR,rrex,rrey

    ########################################################################
    # solve
    def solve(self,alpha):

        #self.read_data()
        num2 = 0

        #fig = plt.figure(tight_layout=True,dpi=300)
        #ax = plt.axes()

        dx = 0.0
        dy = 0.0
        dxpy = 0.0
        dypx = 0.0

        out1 = []
        out2 = []
        out3 = []
        out4 = []


        #print(self.node[1][0],self.node[1][1])
        nodeb = []
        for i in range(0,len(self.node)):
            nodeb.append( self.rotate(self.node[i][0],self.node[i][1],alpha) )
        xg,yg = self.rotate(self.xg,self.yg,alpha)

        #print(nodeb)
        nodeb = np.array(nodeb)

        # 壁量、剛心位置の計算
        for i in range(0,len(self.name)):

            num1 = num2
            num2 = num1 + self.numdata[i]
            #print(num1,num2)
            awx, awy, aw, kx, ky, px, py ,\
                aw_e,awx_e,awy_e,gx_e,gy_e = \
                self.cal(nodeb[num1:num2],self.thn[i])
            #self.cal(self.node[num1:num2],self.thn[i],alpha)

            dx = dx + awx
            dy = dy + awy
            dxpy = dxpy + awx*py
            dypx = dypx + awy*px

            out1.append(awx)
            out2.append(awy)
            out3.append(px)
            out4.append(py)

        ppx = dypx/dy
        ppy = dxpy/dx

        # ねじれ剛性の算定
        # rotational stiffness calculation
        dr = 0.0
        for i in range(0,len(self.name)):

            xb = out3[i] - ppx
            yb = out4[i] - ppy

            dr = dr + out1[i] * yb**2 + out2[i] * xb**2

        #
        # 弾力半径の算定
        rex = math.sqrt(dr/dx)
        rey = math.sqrt(dr/dy)

        #
        # 偏心距離の算定

        if self.xg != -99:
            ex = abs(xg-ppx)
            ey = abs(yg-ppy)
        else:
            ex = 0.0
            ey = 0.0

        # eccentricity ratio
        rex_ratio = ey/rex
        rey_ratio = ex/rey

        # data making
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

        print("alpha=",alpha*360.0/2.0/math.pi,"deg")
        print(df_out)

        print("Total.",\
              "Awx={:10.2e}".format(dx),\
              "Awy={:10.2e}".format(dy),\
              "px={:10.2e}".format(ppx),\
              "py={:10.2e}".format(ppy),\
              "dr={:10.2e}".format(dr)\
              )
        print(\
              "ex={:10.2e}".format(ex),\
              "ey={:10.2e}".format(ey),\
              "rex={:10.2e}".format(rex),\
              "rey={:10.2e}".format(rey),\
              "Rex={:10.2e}".format(rex_ratio),\
              "Rey={:10.2e}".format(rey_ratio)\
              )

        #ax.scatter(ppx,ppy,color='r',marker='D')
        #plt.show()
        #plt.close()
        #self.screen.draw()
        #self.viewModel(self.node[num1:num2],self.thn[i])

        #self.model()
        return dx,dy,ppx,ppy,dr,rex_ratio,rey_ratio

    ########################################################################
    # calculate aw, awx, awy at each element
    ########################################################################
    def cal(self,node,t):

        # node: node data [mm]
        # t   : wall thickness [mm]
        # x & y cordinate from node data

        x = []
        y = []
        for i in range(0,len(node)):
            x.append(node[i,0])
            y.append(node[i,1])

        # initial
        awx = []
        awy = []
        aw = []

        kx = []
        ky = []

        kxgy = []
        kygx = []

        gx = []
        gy = []

        #cs_al = math.cos(alpha)
        #sn_al = math.sin(alpha)

        for i in range(0,len(node)-1):

            l = (x[i+1]-x[i])**2 + (y[i+1]-y[i])**2
            l = math.sqrt(l)

            cs = (x[i+1]-x[i])/l
            sn = (y[i+1]-y[i])/l

            cs2 = cs**2
            sn2 = sn**2

            aw.append( l*t )
            awx.append( l* cs2 * t )
            awy.append( l* sn2 * t )
            gx.append( (x[i] + x[i+1])/2 )
            gy.append( (y[i] + y[i+1])/2 )
            kx.append( t * cs2 )
            ky.append( t * sn2 )
            kxgy.append( t * cs2 * gy[i] )
            kygx.append( t * sn2 * gx[i] )

            """
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

            print("aw,awx,awy,gx,gy",l*t,awx,awy,gx,gy)
            """


        aw  = np.array(aw)
        awx = np.array(awx)
        awy = np.array(awy)
        gx  = np.array(gx)
        gy  = np.array(gy)
        kx  = np.array(kx)
        ky  = np.array(ky)
        kxgy= np.array(kxgy)
        kygx= np.array(kygx)

        if np.sum(ky) == 0 :
            px = np.sum( gx * t * l) / np.sum(aw)
        else:
            px = np.sum(kygx)/np.sum(ky)

        if np.sum(kx) == 0:
            py = np.sum( gy * t * l) / np.sum(aw)
        else:
            py = np.sum(kxgy)/np.sum(kx)

        #return awx, awy, aw, kx, ky, px, py
        return np.sum(awx),np.sum(awy),np.sum(aw),np.sum(kx),np.sum(ky),px,py,\
            aw,awx,awy,gx,gy

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

        allnode = []

        count = 0

        self.xg = -99
        self.yg = -99
        self.reqA = -99

        for i in range(0,num):
            # read parameter
            if(df.iloc[i,0] == "GRAVITY"):
                self.xg = df.iloc[i,1]
                self.yg = df.iloc[i,2]

            if(df.iloc[i,0] == "DEMAND"):
                self.reqA = df.iloc[i,1]

            if(df.iloc[i,0] == "CNTL"):
                name.append( df.iloc[i,1] )
                thn.append( df.iloc[i,2] )

                count = 0

                """
                node = []
                x =[]
                y =[]
                node.clear
                x.clear
                y.clear
                """

            if(df.iloc[i,0] == "NODE"):
                #print("count",count)
                #print("hello")
                allnode.append( (float(df.iloc[i,1]), float(df.iloc[i,2]) ) )
                #x.append( (float(df.iloc[i,1]) ) )
                #y.append( (float(df.iloc[i,2]) ) )
                #dia.append( str(data[6]).replace(' ','') )
                count = count + 1

            if(df.iloc[i,0] == "END"):
                #numdata.append(len(node))
                numdata.append(count)
                """
                if kk == 0:
                    allnode = np.array(node)
                    #allx = np.array(x)
                else:
                    allnode = np.vstack( (allnode, np.array(node)))
                    #allx = np.vstack( (allx, np.array(x))  )
                """

                kk = kk + 1
        print("Finish to read!")

        #print(name,thn)
        #print(numdata)
        #print(allnode)
        #print(allnode[numdata[0]:numdata[1]])

        self.name    = name
        self.thn     = thn
        self.numdata = numdata
        self.node = np.array( allnode )
        #print(node[1][0])

"""        
obj=rout_1()
obj.read_data()
obj.solve()
"""        

