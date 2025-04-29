import math
import numpy as np
def zernike(n,m):
    return int((n*(n+2)+m)/2)
'''
def calc_M_J45_J180(z_list,radius):
    M = -4*math.sqrt(3)*z_list[zernike(2,0)]
    J45 = -2*math.sqrt(6)*z_list[zernike(2,-2)]
    J180 = -2*math.sqrt(6)*z_list[zernike(2,2)]
    return M,J45,J180
'''
#修正公式,输入是角度,计算M,J45,J180
def calc_M_J45_J180(z_list,radius,alpha,phy):
    alpha = math.radians(alpha)
    phy = math.radians(phy)

    M = -(2*math.sqrt(3)*z_list[zernike(2,0)]*(math.pow(math.cos(phy),2)+1) + 
          math.sqrt(6)*z_list[zernike(2,-2)]*math.sin(2*alpha)*math.pow(math.sin(phy),2) + 
          math.sqrt(6)*z_list[zernike(2,2)]*math.cos(2*alpha)*math.pow(math.sin(phy),2)
          )/math.pow((radius*math.cos(phy)),2)
    
    J45 = - (2*math.sqrt(3)*z_list[zernike(2,0)]*math.sin(2*alpha)*math.pow(math.sin(phy),2)+
             math.sqrt(6)*z_list[zernike(2,-2)]*(2*math.pow(math.cos(2*alpha),2)*math.cos(phy)+ math.pow(math.sin(2*alpha),2)*(1+math.pow(math.cos(phy),2)))+
             math.sqrt(6)*z_list[zernike(2,2)]*math.cos(2*alpha)*math.sin(2*alpha)*math.pow((1+math.cos(phy)),2)
             )/math.pow((radius*math.cos(phy)),2)
    
    J180 = - (2*math.sqrt(3)*z_list[zernike(2,0)]*math.cos(2*alpha)*math.pow(math.sin(phy),2)+
             math.sqrt(6)*z_list[zernike(2,-2)]*math.cos(2*alpha)*math.sin(2*alpha)*math.pow((1+math.cos(phy)),2)+
             math.sqrt(6)*z_list[zernike(2,2)]*(math.pow(math.cos(2*alpha),2)*(math.pow(math.cos(phy),2)+1)+2*math.pow(math.sin(2*alpha),2)*math.cos(phy))
             )/math.pow((radius*math.cos(phy)),2)
    

    return M,J45, J180

#Thibos转换
def algo_Thibos_line(z_list,k):
    c_list = []
    c_list.append(z_list[zernike(0,0)])
    c_list.append(z_list[zernike(1,-1)])
    c_list.append(z_list[zernike(1,1)])

    
    c_list.append( (k)*z_list[zernike(2,-2)])
    c_list.append( (k*k+1)*z_list[zernike(2,0)]/2 + math.sqrt(2)*(k*k-1)*z_list[zernike(2,2)]/4)
    c_list.append( math.sqrt(2)*(k*k-1)*z_list[zernike(2,0)]/2 + (k*k+1)*z_list[zernike(2,2)]/2)

    return c_list

def algo_Thibos_mat(z_list,k):
    clist = np.zeros_like(z_list)
    M = [
        [1,0,0,0,-math.sqrt(3),0],
        [0,0,2,0,0,0],
        [0,2,0,0,0,0],
        [0,0,0,0,2*math.sqrt(3),math.sqrt(6)],
        [0,0,0,2*math.sqrt(6),0,0],
        [0,0,0,0,2*math.sqrt(3),-math.sqrt(6)]
    ]

    N = [
        [1,0,0,0,-math.sqrt(3),0],
        [0,0,2*k,0,0,0],
        [0,2,0,0,0,0],
        [0,0,0,0,2*math.sqrt(3)*k*k,math.sqrt(6)*k*k],
        [0,0,0,2*math.sqrt(6)*k,0,0],
        [0,0,0,0,2*math.sqrt(3),-math.sqrt(6)]
    ]

    clist = np.dot(np.dot(np.linalg.inv(np.array(N)),np.array(M)),np.transpose(z_list))
    return np.transpose(clist).tolist()


def calc_sph_cyl_theta(M,J45,J180):
    cyl = -2*math.sqrt(J180*J180+J45*J45)
    sph = M - cyl/2
    theta = math.degrees(math.atan2(J45,J180)/2) #角度
    if J180 < 0:
        theta = theta + 90
    
    if J180 >= 0 and J45 <= 0:
        theta = theta + 180

    return cyl,sph,theta

#复制compute_D_matrix矩阵
def calc_D_matrix(coordinates,n):
    matrix = []
    for x, y in coordinates:                  #X方向
        if n == 0:
            matrix.append([
                1 * 0
            ])
        elif n == 1:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1
            ])
        elif n == 2:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,
                math.sqrt(6) * 2 * y,
                math.sqrt(3) * 4 * x,
                math.sqrt(6) * 2 * x
            ])
        elif n == 3:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,
                math.sqrt(6) * 2 * y,
                math.sqrt(3) * 4 * x,
                math.sqrt(6) * 2 * x,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * (-2 + 9 * x ** 2 + 3 * y ** 2),
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2)
            ])
        elif n == 4:
            matrix.append([
                1 * 0,
                2 * 0,
                2 * 1,   #1
                math.sqrt(6) * 2 * y,
                math.sqrt(3) * 4 * x,
                math.sqrt(6) * 2 * x,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * (-2 + 9 * x ** 2 + 3 * y ** 2),
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                math.sqrt(10) * (12 * x ** 2 * y - 4 * y ** 3),
                math.sqrt(10) * (-6 * y + 24 * x ** 2 * y + 8 * y ** 3),
                math.sqrt(5) * (-12 * x + 24 * x ** 3 + 24 * x * y ** 2),
                math.sqrt(10) * (-6 * x + 16 * x ** 3),
                math.sqrt(10) * (4 * x ** 3 - 12 * x * y ** 2)
            ])
    for x, y in coordinates:              #Y方向
        if n == 0:                        #1
            matrix.append([
                1 * 0
            ])
        elif n == 1:                      #3
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0
            ])
        elif n == 2:                      #6
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0,
                math.sqrt(6) * 2 * x, 
                math.sqrt(3) * 4 * y,
                math.sqrt(6) * -2 * y
            ])
        elif n == 3:                      #10
            matrix.append([
                1 * 0,
                2 * 1,
                2 * 0,
                math.sqrt(6) * 2 * x,
                math.sqrt(3) * 4 * y,
                math.sqrt(6) * -2 * y,
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                math.sqrt(8) * (-2 + 3 * x ** 2 + 9 * y ** 2),
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * -6 * x * y
            ])
        elif n == 4:                       #15
            matrix.append([
                1 * 0,
                2 * 1,    #1
                2 * 0,
                math.sqrt(6) * 2 * x,
                math.sqrt(3) * 4 * y,
                math.sqrt(6) * -2 * y,
                math.sqrt(8) * (3 * x ** 2 - 3 * y ** 2),
                math.sqrt(8) * (-2 + 3 * x ** 2 + 9 * y ** 2),
                math.sqrt(8) * 6 * x * y,
                math.sqrt(8) * -6 * x * y,
                math.sqrt(10) * (4 * x ** 3 - 12 * x * y ** 2),
                math.sqrt(10) * (-6 * x + 8 * x ** 3 + 24 * x * y ** 2),
                math.sqrt(5) * (-12 * y + 24 * x ** 2 * y + 24 * y ** 3),
                math.sqrt(10) * (6 * y - 16 * y ** 3),
                math.sqrt(10) * (-12 * x ** 2 * y + 4 * y ** 3)
            ])
    return np.array(matrix)       #把一行的列表导入数组

#计算放大倍率，Mx = sph 轴向，My = cyl+sph径向
def calc_Mx_My(cyl,sph,input):
    return sph/input, (cyl+sph)/input

#替换函数read_function，对标准点进行Mx,My变换
def calc_std_spots(standard_spots_filepath,Mx=1.0,My=1.0):
    with open(standard_spots_filepath, 'r') as file:
        content = file.read()
    central = eval(content)
    result = [[row[0] * Mx, row[1] * My] for row in central]
    return result

#替换函数getcentraldiffs,对diffx和diffy进行变换
def calc_xydiff(non_standard_spots,list_standard_spots,Mx = 1.0,My = 1.0):
    # 计算偏移量并输出偏移量的结果
    x_diffs = []
    y_diffs = []

    for (x1, y1), (x2, y2) in zip(non_standard_spots, list_standard_spots):
        x_diff = (x1 - Mx*x2)/(Mx*Mx)
        y_diff = (y1 - My*y2)/(My*My)
        x_diffs.append(x_diff)  # 存放x的偏移量
        y_diffs.append(y_diff)  # 存放y的偏移量

    xy_diffs = []
    xy_diffs.extend(x_diffs)
    xy_diffs.extend(y_diffs)
    return xy_diffs,list_standard_spots

#J = D*C
#替换函数averageslope，计算J
def calc_J(xy_diffs,radius,F,X):
    j=[]
    for i in xy_diffs:
        x=i*X #像素转距离，单位um
        f=F*1000 #单位转换，单位um
        r=radius*X #单位转换，单位um
        j.extend([(x/f)]) 
    # J矩阵
    J = np.array(j)
    return J

#替换函数getstandard_spots,计算D
def calc_D(no_standard_points,radius,n,center,Mx,My):
    # #定义归一化后的R列表
    R = []
    # 计算图像中心坐标
    zx = center
    for central in no_standard_points:          #遍历与非标准点对应的标准点坐标
        x = (central[0]/Mx-zx[0])/radius
        y = (central[1]/My-zx[1])/radius
        R.append([x,y])


    coordinates = R

    # 计算D矩阵
    D = calc_D_matrix(coordinates,n)
    return D
