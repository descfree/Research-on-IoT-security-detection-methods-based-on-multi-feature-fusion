from tkinter import *
import tkinter.messagebox
import time
from tkinter import ttk
from tkinter import filedialog
import  csv
from sklearn.metrics import roc_curve, auc
from tkintertable import TableCanvas, TableModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             precision_score, auc)
from sklearn import metrics
import numpy as np
from keras.models import load_model
from kdd_processing import kdd_encoding
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
params = {'train_data': 17000, 'features_nb': 4,
          'batch_size': 1024, 'encoder': 'standarscaler',
          'dataset': 'kdd'}
x_train, x_test, y_train, y_test = kdd_encoding(params)
x_train = np.array(x_train).reshape([-1, x_train.shape[1], 1])
x_test = np.array(x_test).reshape([-1, x_test.shape[1], 1])
window = Tk()
window .geometry('900x700')
window .title('RNN_AUC')


def lct(params, model, x_train, x_test, y_train, y_test):
    y_pred = model.predict(x_test, params['batch_size'])

    conf_matrix = confusion_matrix(y_test.argmax(axis=1),
                                       y_pred.argmax(axis=1))
    fpr, tpr, thersholds = roc_curve(y_test.argmax(axis=1), y_pred.argmax(axis=1), pos_label=1)
    f = Figure(figsize=(5, 5), dpi=100)
    a = f.add_subplot(111)
    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP,
                                fill=X,
                                expand=tkinter.YES)
    AUC = roc_auc_score(y_true=y_test, y_score=y_pred, average=None)
    a.plot(fpr, tpr, label='ROC '.format(AUC), lw=2)
    a.set_xlabel('False Positive Rate')
    a.set_ylabel('True Positive Rate')
    a.set_title('ROC Curve')
    now = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    plt.plot(fpr, tpr, label='ROC '.format(AUC), lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("../ROC/"+now+"_GRU.jpg")
    return fpr, tpr, thersholds

select_path = StringVar()
def select_file():

    selected_file_path = filedialog.askopenfilename()
    select_path.set(selected_file_path)
def inst():
    g = ""
    g = en1.get()
    model_name = g

    if (model_name == ""):
        tkinter.messagebox.showinfo(title='警告', message='未选择模型')
    else:

        print(model_name)
        lct(params, load_model(model_name), x_train, x_test, y_train, y_test)
lb1 = Label(window, text="文件路径：")
lb1.place(relx=0.01, rely=0.8, relwidth=0.1, relheight=0.1)
en1 = Entry(window, textvariable=select_path)
en1.place(relx=0.09, rely=0.8, relwidth=0.2, relheight=0.08)
btn0 = Button(window, text="选择单个文件", command=select_file)
btn0.place(relx=0.4, rely=0.8, relwidth=0.2, relheight=0.1)
btn6 = Button(window, text="生成图", command=inst)
btn6.place(relx=0.7, rely=0.8, relwidth=0.2, relheight=0.1)
window.mainloop()