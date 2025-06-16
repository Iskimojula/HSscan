import optometry as opt
import csv
import math

mode = "THIBOS"
i = 0
def exit_pupil_2_entrence_pupil(zlist_exit,zlist_entrence,Ms,Mt,alpha,phy,radius_input,sph_input,radius_output):
    clist_i = opt.algo_elli_correction(zlist_exit,Ms,Mt,radius_input,radius_output)
    if 0:
        M,J45, J180 = opt.calc_M_J45_J180(clist_i,radius_input,alpha,phy)

        a3 = J45*radius_input*radius_input/(-2*math.sqrt(6))
        a4 = M*radius_input*radius_input/(-4*math.sqrt(3))
        
        a5 = J180*radius_input*radius_input/(-2*math.sqrt(6))

    M,J45, J180 = opt.calc_M_J45_J180_noatchson(clist_i,radius_input)
    cyl,sph,theta = opt.calc_sph_cyl_theta(M,J45,J180)
    if 1:
        print(f"{i},{sph_input},{clist_i[3]:.3f},{clist_i[4]:.3f},{clist_i[5]:.3f},{cyl:.3f},{sph:.3f},{theta:.3f}")
    if 0:
        print('')
        print("input sph:",sph_input )
        print("exit z:",zlist_exit[3],zlist_exit[4],zlist_exit[5])
        print(" correction elli entrance pupil z:",clist_i[3],clist_i[4],clist_i[5])
        print(" atchson correction elli entrance pupil z:",a3,a4,a5)
        print(" simudata entrance:",zlist_entrence[3],zlist_entrence[4],zlist_entrence[5])
        #print("calc after elli_Atchison correction M, J45, J180:",M, J45, J180)
        print("calc after elli_Atchison correction cyl,sph,theta:",cyl,sph,theta)
        print('')



def exit_pupil_2_entrence_pupil_THIBOS(zlist_exit,zlist_entrence,Ms,Mt,alpha,phy,radius_input,sph_input,radius_output):
    k = math.sqrt(Mt/Ms)
    clist_i = opt.algo_Thibos_mat(zlist_exit,k)

    M,J45, J180 = opt.calc_M_J45_J180(clist_i,radius_input,alpha,phy)

    a3 = J45*radius_input*radius_input/(-2*math.sqrt(6))
    a4 = M*radius_input*radius_input/(-4*math.sqrt(3))
    
    a5 = J180*radius_input*radius_input/(-2*math.sqrt(6))

    cyl,sph,theta = opt.calc_sph_cyl_theta(M,J45,J180)
    print('')
    print("input sph:",sph_input )
    print("exit z:",zlist_exit[3],zlist_exit[4],zlist_exit[5])
    print(" correction elli entrance pupil z:",clist_i[3],clist_i[4],clist_i[5])
    print(" atchson correction elli entrance pupil z:",a3,a4,a5)
    print(" simudata entrance:",zlist_entrence[3],zlist_entrence[4],zlist_entrence[5])
    #print("calc after elli_Atchison correction M, J45, J180:",M, J45, J180)
    print("calc after elli_Atchison correction cyl,sph,theta:",cyl,sph,theta)
    print('')

PJsimudata = 'ABSsimudataall.csv'


alpha = 0
phy = -12

with open(PJsimudata,'r',encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for line in reader:
        i +=1
        zlist_entrence = list(map(float,line[:6]))
        zlist_exit = list(map(float,line[6:12]))

        radius_output = float(line[-1])/2
        radius_input = float(line[-2])/2
        sph_input = float(line[-3])
        Ms = min(math.sqrt(math.sqrt(float(line[-4])*float(line[-4]))),math.sqrt(math.sqrt(float(line[-5])*float(line[-5]))))
        Mt = max(math.sqrt(math.sqrt(float(line[-4])*float(line[-4]))),math.sqrt(math.sqrt(float(line[-5])*float(line[-5]))))

        exit_pupil_2_entrence_pupil(zlist_exit,zlist_entrence,Ms,Mt,alpha,phy,radius_input,sph_input,radius_output)







