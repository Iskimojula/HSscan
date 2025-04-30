import optometry as opt

zlist = [-7.405,0.0,-0.505,0.0,-4.881,-0.366]

k = 1/1.048
radius = 2
alpha = 0
phy = -15

#Thibos算法修正
clist = opt.algo_Thibos_line(zlist,k)
clist_m = opt.algo_Thibos_mat(zlist,(1/k))

clist_i = clist.copy()
clist_r = [-10.983,0.00000000,-0.072, 0.0,-4.881,-0.366] 


M,J45, J180 = opt.calc_M_J45_J180(clist_i,radius,alpha,phy)

cyl,sph,theta = opt.calc_sph_cyl_theta(M,J45,J180)

M_r,J45_r, J180_r = opt.calc_M_J45_J180(clist_r,radius,alpha,phy)

print("exit z:",zlist)

print("thibos correction entrance pupil z:",clist_i)
print("simu entrance pupil z:",clist_r )
print("thibos correction entrance pupil z in mat", clist_m)
print("calc after elli_Atchison correction M, J45, J180:",M, J45, J180)
print("calc after elli_Atchison correction cyl,sph,theta:",cyl,sph,theta)

print("simu after elli_Atchison correction M, J45, J180:",M_r, J45_r, J180_r)

