import csv
import numpy as np
from matplotlib import pyplot as plt
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
col_types = [float, float, float, float, float, float, float]
window = Tk()
window .geometry('900x700')
window .title('LOSS')

with open('../results/testcsv/full_results.csv') as f:
    f_csv = csv.reader(f)

    headers = next(f_csv)

    enc_x, acc_y ,y_train_loss = [], [],[]

    index = 0

    for row in f_csv:
        row = tuple(convert(value) for convert, value in zip(col_types, row))
        enc_x.append(row[0])
        acc_y.append(row[1])
        y_train_loss.append(row[2])
        x_train_loss=range(len(y_train_loss))
        index += 1
    f = Figure(figsize=(5, 5), dpi=100)
    a = f.add_subplot(111)
    canvas = FigureCanvasTkAgg(f, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tkinter.TOP,
                                fill=BOTH,
                                expand=tkinter.YES)

    a.plot(x_train_loss, y_train_loss, label='LOSS ', lw=2)
    a.set_xlabel('iters')
    a.set_ylabel('loss')
    a.set_title('LOSS')

    plt.plot(x_train_loss,  y_train_loss, color="green", label="angular_x")
    plt.axis([0,300,0,0.01])
    plt.xlabel('iters')
    plt.ylabel('loss')
    now = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    plt.savefig("../LOSS/"+now+"LOSS.jpg")

    plt.show()

window.mainloop()