import matplotlib.pyplot as plt
import numpy as np
 



x_axis_data = [1,5,10,20,50,100,200]
y_dist20_patch = [0.2043,0.3290,0.3545,0.3622,0.3697,0.3355,0.3314]
y_dist20_swin = [0.2043,0.5629,0.5806,0.5987,0.6259,0.6021,0.5903]
y_dist50_patch = [0.6200,0.8023,0.8266,0.8468,0.8504,0.8456,0.8325]
y_dist50_swin = [0.6200,0.8634,0.9030,0.9231,0.9448,0.9293,0.9086]

        
#画图 

plt.plot(x_axis_data, y_dist20_patch, 'go--', alpha=0.5, linewidth=1, label='Patch-NetVLAD-2048 dist20')#'
plt.plot(x_axis_data, y_dist20_swin, 'ms-', alpha=0.5, linewidth=1, label='CurriculumLoc(ours) dist20')
plt.plot(x_axis_data, y_dist50_patch, 'bo--', alpha=0.5, linewidth=1, label='Patch-NetVLAD-2048 dist50')
plt.plot(x_axis_data, y_dist50_swin, 'rs-', alpha=0.5, linewidth=1, label='CurriculumLoc(ours) dist50')
plt.xticks([1,5,10,20,50,100,200])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])

for a, b in zip(x_axis_data, y_dist20_patch):
    plt.text(a, b, str(b), ha='center', va='bottom', fontsize=8)  #  ha='center', va='top'
for a, b1 in zip(x_axis_data, y_dist20_swin):
    plt.text(a, b1, str(b1), ha='center', va='bottom', fontsize=8)  
for a, b2 in zip(x_axis_data, y_dist50_patch):
    plt.text(a, b2, str(b2), ha='center', va='bottom', fontsize=8)
for a, b3 in zip(x_axis_data, y_dist50_swin):
    plt.text(a, b3, str(b3), ha='center', va='bottom', fontsize=8)

 
plt.legend()  #显示上面的label
plt.xlabel('Candidates number')
plt.ylabel('rerank R@1')#accuracy
 
 
plt.savefig("./plot_rs.png")
plt.show()

