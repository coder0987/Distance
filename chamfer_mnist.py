import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from enum import Enum

import time
from py4j.java_gateway import JavaGateway

gateway = JavaGateway()
java_distance = gateway.entry_point

class DistanceFunction(Enum):
    EUCLIDEAN = 0
    CHAMFER = 1
    CHAMFER_TRANSFORMED = 2
    DTW = 3

DISTANCE_FUNCTION = DistanceFunction.DTW
LOAD_IMGS = True
RESET_JAVA = True # MUST BE TRUE IF M/N ARE ADJUSTED
K = 1

N = 60000 # of training examples, up to 60k
M = 10000 # of test cases, up to 10k

IMG_SIZE = 784 # 28 ** 2

if LOAD_IMGS:
    dataset = tf.keras.datasets.mnist
    (training_set, test_set) = dataset.load_data()
    (training_images, training_labels) = training_set
    (test_images, test_labels) = test_set

    print("training images shape:", training_images.shape, "")
    print("training images dtype:", training_images.dtype, "")
    print("training labels shape:", training_labels.shape, "")

    print("test images shape:", test_images.shape, "")
    print("test images dtype:", test_images.dtype, "")
    print("test labels shape:", test_labels.shape, "")



    #%%

    training_index = 17
    example_image = training_images[training_index]
    label = training_labels[training_index]
    plt.imshow(example_image, cmap='gray')
    print("dtype:", example_image.dtype)
    print("shape:", example_image.shape)
    print("class label:", label)




#%%
"""
(training_number, rows, cols) = training_images.shape
(test_number, rows, cols) = test_images.shape

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(rows, cols)),
    tf.keras.layers.Dense(512, activation='tanh'),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

training_inputs = training_images / 255.0
test_inputs = test_images / 255.0

model.fit(training_inputs, training_labels, epochs=15)
test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)

print('\nTest accuracy: %.2f%%' % (test_acc * 100))
"""

def shortest_distance(point, points):
    #finds the shortest Euclidean distance between a point and an array of points (x, y) and arr[(x,y),...]
    return np.minimum.reduce(
        np.sqrt(
            np.add(
                np.square(
                    np.subtract(point[0], points.T[0])
                ),
                np.square(
                    np.subtract(point[1], points.T[1])
                )
            )
        )
    )

def directed_chamfer_distance(active1, active2):
    #arrays of tuples (x, y)
    dist = 0
    for el in active1:
        dist += shortest_distance(el, active2)

    dist /= len(active1)

    return dist

def chamfer_distance(image1, image2):
    active1 = np.transpose(np.nonzero(image1))
    active2 = np.transpose(np.nonzero(image2))
    return directed_chamfer_distance(active1, active2) + directed_chamfer_distance(active2, active1)

def chamfer_distance_transformed(image1, image2):
    image1 = np.array(image1, dtype=np.int16) # convert to signed data type
    image2 = np.array(image2, dtype=np.int16)
    diff = image1 - image2
    diff2 = image2 - image1
    return np.sum(
        np.absolute(diff) + np.absolute(diff2)
    )



def euclidean_distance(image1, image2):
    sumImg = image1 + image2 - 1 #-1 or 1 if they are different, 0 if they are the same
    return IMG_SIZE - np.count_nonzero(sumImg) #no countzero function, so take the size and subtract the nonzeros
    
    #return np.sum(sumImg, where=(sumImg==1)) #old method, which doesn't include the -1 for sumImg. Half the speed

def swim_up(heap, i):
    if (i == 0):
        return
    elif (heap[i]["distance"] > heap[math.floor(i/2)]["distance"]):
        t = heap[i]
        heap[i] = heap[math.floor(i/2)]
        heap[math.floor(i/2)] = t
        swim_up(heap, math.floor(i/2))



def knn_classify(image1, training_set, training_labels, k):
    nearest = [] # will be {label: string, distance: float}. Should only hold k nearest

    for i,img in enumerate(training_set):
        nearest.append({"label": training_labels[i], "distance": dist_func(image1, img)})
        swim_up(nearest, len(nearest) - 1)
        if (len(nearest) > k):
            nearest.pop(0)
        #print(nearest)

    top_k = {}
    top = nearest[0]["label"]
    most = 0
    for el in nearest:
        if el["label"] not in top_k:
            top_k[el["label"]] = 1
        else:
            top_k[el["label"]] += 1
        
        if (top_k[el["label"]] > most):
            most = top_k[el["label"]]
            top = el["label"]
        
        #print("Num [%d] distance [%d]" % (el["label"], el["distance"]))

    return top

def knn_evaluate(test_set, test_labels, training_set, training_labels, k):
    correct = 0
    incorrect = 0

    for i,img in enumerate(test_set):
        classification = knn_classify(img, training_set, training_labels, k)
        if (classification == test_labels[i]):
            correct+=1
        else:
            incorrect+=1
        print("Test image %5d, k=%2d, knn estimated label: %d, true label %d" % (i, k, classification, test_labels[i]))
    
    print("Classified correctly %5d out of %5d images, accuracy = %.2f%%" % (correct, correct + incorrect, (correct / (correct + incorrect) * 100)))

#%%

"""Turns all nonzero values to 1"""
def preprocess(arr):
    arr[np.nonzero(arr)] = 1
    return arr

def print_time(start, end, msg):
    elapsed_time = end - start
    minutes = elapsed_time // 60
    seconds = elapsed_time % 60
    milliseconds = (elapsed_time * 1000) % 1000
    print("%s finished in %f min, %f sec, %f ms" % (msg, minutes, seconds, milliseconds))



pre_start = time.time()

if RESET_JAVA:
    java_distance.clear() # Delete all image data and reload it
    pre_imgs = preprocess(test_images[0:M:1])
    pre_train_imgs = preprocess(training_images[0:N:1])

    labels = test_labels[0:M:1]
    train_labels = training_labels[0:N:1]

    # Send arrays over to Java
    java_distance.loadTestData(pre_imgs.tobytes())
    java_distance.loadTrainingData(pre_train_imgs.tobytes())

if LOAD_IMGS and DISTANCE_FUNCTION == DistanceFunction.EUCLIDEAN:
    pre_imgs = preprocess(test_images[0:M:1])
    pre_train_imgs = preprocess(training_images[0:N:1])

    labels = test_labels[0:M:1]
    train_labels = training_labels[0:N:1]
elif LOAD_IMGS:
    labels = test_labels[0:M:1]
    train_labels = training_labels[0:N:1]

#distance_transform_imgs = np.frombuffer(java_distance.distance_transform(), dtype=np.uint8)
#distance_transform_training_imgs = np.frombuffer(java_distance.distance_transform_training(), dtype=np.uint8)

""" DEBUG
if not LOAD_IMGS:
    java_distance.debug(1)
    java_distance.debug(2)
"""
    
dist_func = euclidean_distance #default to euclidean

# get latest img data from Java JVM
if DISTANCE_FUNCTION == DistanceFunction.CHAMFER_TRANSFORMED:
    distance_transform_imgs = np.reshape(np.frombuffer(java_distance.distance_transform(), dtype=np.uint8), (M, 28, 28))
    distance_transform_training_imgs = np.reshape(np.frombuffer(java_distance.distance_transform_training(), dtype=np.uint8), (N, 28, 28))
    dist_func = chamfer_distance_transformed
elif DISTANCE_FUNCTION == DistanceFunction.DTW:
    java_distance.edge_transform_test_images()
    java_distance.edge_transform_training_images()
    java_distance.loadTestLabels(labels.tobytes())
    java_distance.loadTrainingLabels(train_labels.tobytes())
else:
    distance_transform_imgs = pre_imgs
    distance_transform_training_imgs = pre_train_imgs

if DISTANCE_FUNCTION == DistanceFunction.CHAMFER:
    dist_func = chamfer_distance

print_time(pre_start, time.time(), "Preprocessing")


time.sleep(20)

start = time.time()

#debug


#print(distance_transform_imgs[0])
if DISTANCE_FUNCTION == DistanceFunction.DTW:
    java_distance.knn_evaluate(K)
elif LOAD_IMGS:
    knn_evaluate(distance_transform_imgs, labels, distance_transform_training_imgs, train_labels, K)

print_time(start, time.time(), "Evaluation")
