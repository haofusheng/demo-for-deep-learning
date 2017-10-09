from matplotlib import pyplot as plt
import numpy as np

root = 'E:\\program\\python\\center_bias\\'
log = 'log.txt'

max_iteration = 320000
test_interval = 1000
epoches = 64.0

# loss and accuracy
train_loss = []
train_accuracy = []
val_loss = []
val_accuracy = []

# locators
trainSeparator = 'Train net output #'
valSeparator = 'Test net output #'
lossSeparator = ' loss = '
accuracySeparator = ' accuracy = '
losslocator = ' (*'

with open(root + log) as f:
    L = f.readline()
    while L:
        t_p = L.find(trainSeparator)
        a_p = L.find(valSeparator)
        if t_p == -1 and a_p == -1:
            L = f.readline()
            continue
        elif t_p != -1:
            L = L[t_p + len(trainSeparator):]
            t_l_p = L.find(lossSeparator) 
            t_a_p = L.find(accuracySeparator)
            if t_a_p == -1 and t_l_p == -1:
                L = f.readline()
                continue
            if t_l_p != -1:
                L = L[t_l_p + len(lossSeparator):]
                l_p = L.find(losslocator)
                if l_p == -1:
                    L = f.readline()
                    continue
                train_loss.append(float(L[:l_p]))
            if t_a_p != -1:
                L = L[t_a_p + len(accuracySeparator):]
                train_accuracy.append(float(L))
        elif a_p != -1:
            L = L[a_p + len(valSeparator):]
            v_l_p = L.find(lossSeparator)
            v_a_p = L.find(accuracySeparator)
            if v_l_p == -1 and v_a_p == -1:
                L = f.readline()
                continue
            if v_l_p != -1:
                L = L[v_l_p + len(lossSeparator):]
                l_p = L.find(losslocator)
                if l_p == -1:
                    L = f.readline()
                    continue
                val_loss.append(float(L[:l_p]))
            if v_a_p != -1:
                L = L[v_a_p + len(accuracySeparator):]
                val_accuracy.append(float(L))
                
        L = f.readline()

x = np.arange(0, epoches, epoches* test_interval / max_iteration)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

l1, = ax1.plot(x,train_loss[::50],color='r')
l2, = ax1.plot(x,val_loss,color='b')
ax1.set_xlabel('Epoch',fontsize=14)
ax1.set_ylabel('Loss',fontsize=14)
l3, = ax2.plot(x,val_accuracy,color='g')
ax2.set_ylabel('Top-1 Accuracy',fontsize=14)
ax1.legend((l1,l2,l3),('train loss','test loss','val accuracy'), loc='upper right',fontsize=12)
plt.xlim([0, 10 * int(np.ceil(epoches/10))])
plt.xticks(np.arange(0, 10 * int(np.ceil(epoches/10) + 1), 10))
plt.show()
