import optometry as opt

dir = "chuizhi"
angle= -10

aber = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#本征像差
init = opt.get_initaberration(dir,angle)

print(init)
#减去本征像差
result = [x - y for x, y in zip(aber, init)]

print(result)  # 输出: [9, 18, 27]