#coding:utf-8
from  numpy import *
from tkinter import *
import regression_Tree.regTrees
import matplotlib
matplotlib.use('TkAgg') #设置matplotlib后端为TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS , tolN):
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get(): #检查复选框是否选中，确定构造模型树还是回归树
        if tolN < 2 : tolN = 2
        myTree = regression_Tree.regTrees.createTree(reDraw.rawDat , regression_Tree.regTrees.modelLeaf , regression_Tree.regTrees.modelErr,(tolS , tolN))
        yHat = regression_Tree.regTrees.createForeCast(myTree , reDraw.testDat , regression_Tree.regTrees.modelTreeEval)
    else:
        myTree = regression_Tree.regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = regression_Tree.regTrees.createForeCast(myTree, reDraw.testDat)
    # reDraw.rawDat[:,0].A，需要将矩阵转换成数组
    reDraw.a.scatter(reDraw.rawDat[:,0].A,reDraw.rawDat[:,1].A,s=5) #真实值,散点图
    reDraw.a.plot(reDraw.testDat , yHat , linewidth = 2.0) #预测值，连续曲线
    reDraw.canvas.show()

def getInput(): #获取谁
    try:tolN = int(tolNentry.get())
    except:
        tolN = 10
        print("enter Integer for tolN") #确保tolN是整数
        tolNentry.delete(0,END)
        tolNentry.insert(0,'10')
    try:tolS = float(tolSentry.get())
    except:
        tolS = 1.0
        print("enter Float for tolS") #确保tolS是小数
        tolSentry.delete(0,END)
        tolSentry.insert(0,'1.0')
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInput()
    reDraw(tolN,tolS)

if __name__ == "__main__":
    root = Tk() #先创建一个Tk类型的根部件，然后插入标签

    reDraw.f = Figure(figsize=(5,4),dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f , master=root) #画布
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row = 0 , columnspan = 3)

    # 用.grid()方法设定行和列的位置，通过columnspan和rowspan值告诉布局管理器是否
    # 允许一个小部件跨行或跨列
    Label(root, text="Plot Place Holder").grid(row=0, columnspan=3)
    Label(root, text="tolN").grid(row=1, column=0)
    tolNentry = Entry(root) #文本输入框
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    Label(root, text="tolS").grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')

    Button(root, text="ReDraw", command=drawNewTree).grid(row=1, column=2, rowspan=3)
    chkBtnVar = IntVar()  #按钮整数值，为了读取Checkbutton的状态而创建
    chkBtn = Checkbutton(root, text="Model Tree", variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)

    reDraw.rawDat = mat(regression_Tree.regTrees.loadDataSet('sine.txt'))
    reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
    reDraw(1.0, 10)

    root.mainloop() #启动事件循环，使该窗口在众多事件中可以响应鼠标点击、按键和重绘等动作