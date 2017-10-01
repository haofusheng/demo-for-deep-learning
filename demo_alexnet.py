"""

Demo in the form of animation for Alexnet. 

"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sys
import time

# specify the number of test images
num_test = 10000

# display buffer
display = np.zeros((227, 227, 3), np.uint8)

# load caffe
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
import caffe

# GPU mode
caffe.set_device(2)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

# load net
model_def = caffe_root + 'models/bvlc_reference_caffenet/train_val.prototxt'
model_weights = caffe_root + 'bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,  # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)  # use test mode (e.g., don't perform dropout)

# classes
classes = []
Separator1 = ' '
Separator2 = ','
with open(caffe_root + 'data/ilsvrc12/' + 'synset_words.txt') as f:
    line = f.readline()
    while line:
        line = line[line.find(Separator1) + 1:]
        line = line[:line.find(Separator2)]
        classes.append(line)
        line = f.readline()

# net forward
net.forward()

# obtain prob, data, and label
prob = net.blobs['prob'].data[0]
data = net.blobs['data1'].data[0]
label = net.blobs['label'].data[0].astype(np.int)

# initialize display
display[:, :, 0] = data[2].astype(np.uint8)
display[:, :, 1] = data[1].astype(np.uint8)
display[:, :, 2] = data[0].astype(np.uint8)

# sort top one predictions from softmax output
top_inds = prob.argsort()[::-1][:1]

plt.rcParams['figure.figsize'] = (9, 4)
fig = plt.figure()
plt.suptitle('Top one/five accuracy',fontsize=15)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.set_axis_off()
ax2.set_axis_off()

im1 = ax1.imshow(display, animated=True)
ax2.axis([0, 10, 0, 10])
groundtruth = ax2.text(1.5, 4.5, 'Groundtruth: ' + classes[label], fontsize=12)
testtime = ax2.text(1.5, 3.5, 'Test time per image (ms): ', fontsize=10)

if label == top_inds[0]:
    predicted = ax2.text(1.5, 5.5, 'Predicted: {}'.format(classes[top_inds[0]]), fontsize=12, color='green')
else:
    predicted = ax2.text(1.5, 5.5, 'Predicted: {}'.format(classes[top_inds[0]]), fontsize=12, color='red')

def blanklines(n):
    for i in range(n):
        print('\n')

if label == top_inds:
    blanklines(7)
    print('This picture is sorted correctly')
else:
    blanklines(7)
    print('This picture is classified incorrectly')
print('Groundtruth: {} \n Predicted: {}'.format(classes[label], classes[top_inds[0]]))
blanklines(7)

def updatefig(*args):
    start = time.clock()
    net.forward()
    end = time.clock()
    testtime.set_text('Test time per image (ms): {}'.format(str(end - start)))
    prob = net.blobs['prob'].data[0]
    data = net.blobs['data1'].data[0]
    label = net.blobs['label'].data[0].astype(np.int)

    top_inds = prob.argsort()[::-1][:1]

    display[:, :, 0] = data[2].astype(np.uint8)
    display[:, :, 1] = data[1].astype(np.uint8)
    display[:, :, 2] = data[0].astype(np.uint8)

    im1.set_array(display)
    groundtruth.set_text('Groundtruth: ' + classes[label])
    predicted.set_text('Predicted: ' + classes[top_inds[0]])
    if top_inds[0] == label:
        predicted.set_color('green')
    else:
        predicted.set_color('red')
    if label == top_inds[0]:
        blanklines(7)
        print('This picture is sorted correctly')
    else:
        blanklines(7)
        print('This picture is classified incorrectly')
    print('Groundtruth: {} \nPredicted: {} '.format(classes[label], classes[top_inds[0]]))
    blanklines(7)
    return im1,groundtruth,predicted,testtime,

ani = animation.FuncAnimation(fig, updatefig, interval=2000, blit=True)
plt.show()
