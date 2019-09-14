import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_curve(x,y):
  plt.plot(x,y)
  plt.show()

def plot_3d(xs,ys,zs):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(xs,ys,zs,s=2)
#  zeros = [0]*len(self.classes)
#  ax.quiver(zeros,zeros,zeros,xs,ys,zs,arrow_length_ratio=0.01)
#  ax.set_xlim3d(-1, 1)
#  ax.set_ylim3d(-1,1)
#  ax.set_zlim3d(-1,1)
  plt.show()

