import matplotlib.pyplot as plt
fig = plt.figure()
fig.clf()
ax = plt.subplot(111) #用于构建画笔
ax.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
ax.plot([1,2,3], [1,4,9], 'rs',  label='line 2')
ax.axis([0, 4, 0, 10])
ax.legend()
plt.show()
