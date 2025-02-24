import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math



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
        nearest.append({"label": training_labels[i], "distance": chamfer_distance(image1, img)})
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

import time

start = time.time()

knn_evaluate(test_images[650:700:1], test_labels[650:700:1], training_images[0:75:1], training_labels[0:75:1], 5)

elapsed_time = time.time() - start

minutes = elapsed_time // 60
seconds = elapsed_time % 60
milliseconds = (elapsed_time * 1000) % 1000
print("Your code ran in %f min, %f sec, %f ms" % (minutes, seconds, milliseconds))
