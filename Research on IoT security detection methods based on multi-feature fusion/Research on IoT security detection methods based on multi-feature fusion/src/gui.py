from tkinter import *
import tkinter.messagebox
from tkinter import ttk
from tkinter import filedialog
import  csv

from tkintertable import TableCanvas, TableModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)
from sklearn import metrics
import numpy as np
from keras.models import load_model
from kdd_processing import kdd_encoding
from results_visualisation import res
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
params = {'train_data': 17000, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler',
          'dataset': 'kdd','epochs':300}



x_train, x_test, y_train, y_test = kdd_encoding(params)
x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])

root = Tk()
root.geometry('900x400')
root.title('训练模型')
columns=['TPR','FPR','AUC','pre']
table=ttk.Treeview(
    master=root,
    height=10,
    columns=columns,
    show='headings',

)
table.heading('TPR', text='TPR' )
table.heading('FPR', text='FPR' )
table.heading('AUC', text='AUC' )
table.heading('pre', text='precision' )
table.column('TPR',width=150,anchor=CENTER)
table.column('FPR',width=150,anchor=CENTER)
table.column('AUC',width=150,anchor=CENTER)
table.column('pre',width=150,anchor=CENTER)

table.pack(pady=20)








def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
base='D:/RNN_Intrusion-Detection_Keras-master/RNN_Intrusion-Detection_Keras-master/models/'


select_path = StringVar()
def insert():
    for i in findAllFile(base):


       print(i)


       TPR,FPR,AUC,pre = res(params, load_model(i), x_train, x_test, y_train, y_test)
       info = [TPR, FPR, AUC, pre]
       table.insert('', END, values=info)
       with open('../csv/result.csv', 'a', encoding='utf-8', newline='') as f:  # train数据集

           write = csv.writer(f)


           data = [TPR, FPR, AUC, pre,params['epochs']]
           write.writerow(data)
       print(TPR, FPR, AUC, pre)

def goLSTM():
    os.system("python LSTM_AUC.py")
def goRNN():
    os.system("python LSTM_AUC.py")
def goGRU():
    os.system("python LSTM_AUC.py")
def goACC():
    os.system("python ACC.py")
def goLOSS():
    os.system("python LOSS.py")
btn1 = Button(root, text='生成LSTM_AUC',command=goLSTM)
btn1.place(relx=0.75, rely=0.8, relwidth=0.2, relheight=0.1)
btn3 = Button(root, text='生成RNN_AUC',command=goRNN)
btn3.place(relx=0.15, rely=0.65, relwidth=0.2, relheight=0.1)
btn4 = Button(root, text='生成GRU_AUC',command=goGRU)
btn4.place(relx=0.75, rely=0.65, relwidth=0.2, relheight=0.1)
btn2 = Button(root, text='显示结果',command=insert)
btn2.place(relx=0.15, rely=0.8, relwidth=0.2, relheight=0.1)
btn5 = Button(root, text='显示ACC',command=goACC)
btn5.place(relx=0.45, rely=0.8, relwidth=0.2, relheight=0.1)
btn6 = Button(root, text='显示LOSS',command=goLOSS)
btn6.place(relx=0.45, rely=0.65, relwidth=0.2, relheight=0.1)



root.mainloop()