# NOT YOLO
# Epoch 20/20
# 876/876 [==============================] - 2s 3ms/step - loss: 2.2974 - accuracy: 0.3061 - val_loss: 2.6739 - val_accuracy: 0.2347

# YOLO
# Epoch 20/20
# 319/319 [==============================] - 1s 3ms/step - loss: 0.8691 - accuracy: 0.7023 - val_loss: 3.6715 - val_accuracy: 0.3204

import json
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Conv1D, Input, concatenate, MaxPooling1D, Dropout, Reshape, LSTM, TimeDistributed
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from keras.models import Model
from tensorflow.keras.layers import LayerNormalization, Attention
import tensorflow as tf
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import keras.backend as K
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold

def get_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_performance_metrics(y_true, y_pred):
    y_pred_formatted = []
    y_true_formatted = []
    for i in range(len(y_pred)):
        clip_pred = y_pred[i]
        clip_true = y_true[i]
        pred = np.argmax(clip_pred)
        true = np.argmax(clip_true)
        y_pred_formatted.append(pred)
        y_true_formatted.append(true)

    # Step 2: Calculate the Average Precision (AP) for each class
    average_precision_scores = []
    for class_index in range(y_pred.shape[1]):
        ap = average_precision_score(y_true[:, class_index], y_pred[:, class_index])
        average_precision_scores.append(ap)

    # Step 3: Calculate the mean Average Precision (mAP)
    mAP = np.mean(average_precision_scores)
    
    precision = precision_score(y_true_formatted, y_pred_formatted, average='macro')
    recall = recall_score(y_true_formatted, y_pred_formatted, average='macro')
    f1 = f1_score(y_true_formatted, y_pred_formatted, average='macro')
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("mAP: ", mAP)
    return precision, recall, f1, mAP

def calculate_action_performance(y_true, y_pred):
    y_true_count = dict()
    for val in y_true:
        if val in y_true_count.keys():
            y_true_count[val][1] += 1
        else:
            y_true_count[val] = [0, 0]

    for i in range(len(y_true)):
        if y_true[i] == np.argmax(y_pred[i]):
            y_true_count[y_true[i]][0] += 1
    y_pred_percentage = dict()

    for key, val in y_true_count.items():
        y_pred_percentage[key] = val[0]/val[1]
    return y_pred_percentage

def calculate_avg_label_percentage(ls):
    res = dict()
    for i in range(num_action_labels):
        count = 0
        for elem in ls:
            count += elem[i]
        res[i] = round(count/len(ls) * 100, 2)
    return res


x_val_pose = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_pose_yolo.json"))
x_val_confidence = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_confidence_yolo.json"))
x_val_object_classes = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_object_classes_yolo.json"))
x_val_object_boxes = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_val_object_boxes_yolo.json"))
y_val_labels = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\y_val_labels_yolo.json"))
x_train_pose = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_pose_yolo.json"))
x_train_confidence = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_confidence_yolo.json"))
x_train_object_classes = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_object_classes_yolo.json"))
x_train_object_boxes = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\x_train_object_boxes_yolo.json"))
y_train_labels = np.array(get_json_data("D:\\Users\\SMH\\Tim_v2\\data\\y_train_labels_yolo.json"))

pose_data = np.concatenate((x_val_pose, x_train_pose), axis=0)
confidence_data = np.concatenate((x_val_confidence, x_train_confidence), axis=0)
object_classes_data = np.concatenate((x_val_object_classes, x_train_object_classes), axis=0)
object_boxes_data = np.concatenate((x_val_object_boxes, x_train_object_boxes), axis=0)
labels_data = np.concatenate((y_val_labels, y_train_labels), axis=0)

# Define the number of folds
n_splits = 10  # You can adjust this as needed

# Initialize StratifiedKFold
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)

# Initialize lists to store results
poseobj_accuracies = []
poseobj_losses = []
poseobj_precisions = []
poseobj_recalls = []
poseobj_f1s = []
poseobj_maps = []
poseobj_label_percentages = []
pose_accuracies = []
pose_losses = []
pose_precisions = []
pose_recalls = []
pose_f1s = []
pose_maps = []
pose_label_percentages = []


######################################################
# Building the model
######################################################
# Define the input shapes for each of the input types
input_shape_pose = (10, 19, 2)
input_shape_confidence = (10, 19)
input_shape_object_classes = (10, 80)
input_shape_object_boxes = (10, 1, 4)
num_action_labels = 17


def create_model(input_shape_pose, input_shape_confidence, input_shape_object_classes, input_shape_object_boxes, num_action_labels):
    # Define the input layers for each input type
    input_pose = Input(shape=input_shape_pose)
    input_confidence = Input(shape=input_shape_confidence)
    input_object_classes = Input(shape=input_shape_object_classes)
    input_object_boxes = Input(shape=input_shape_object_boxes)
    # Apply LSTM to each time step for human pose
    x = TimeDistributed(LSTM(units=64, return_sequences=True))(input_pose)
    x = Flatten()(x)
    # LSTM for human pose confidence
    y = LSTM(units=64, return_sequences=True)(input_confidence)
    y = Flatten()(y)
    # MLP for object classes
    z = Reshape((10, 80))(input_object_classes)
    z = Flatten()(z)
    # MLP for object boundary boxes
    w = Reshape((10, 4))(input_object_boxes)
    w = Flatten()(w)
    # Concatenate all inputs
    merged = concatenate([x, y, z, w])
    # Fully connected layers for classification
    out = Dense(units=num_action_labels, activation='softmax')(merged)
    # Create the model
    model = Model(inputs=[input_pose, input_confidence, input_object_classes, input_object_boxes], outputs=out)
    return model

def create_base_model(input_shape_pose, input_shape_confidence, num_action_labels):
    # Define the input layers for each input type
    input_pose = Input(shape=input_shape_pose)
    input_confidence = Input(shape=input_shape_confidence)
    # Apply LSTM to each time step for human pose
    x = TimeDistributed(LSTM(units=64, return_sequences=True))(input_pose)
    x = Flatten()(x)
    # LSTM for human pose confidence
    y = LSTM(units=64, return_sequences=True)(input_confidence)
    y = Flatten()(y)
    # Concatenate all inputs
    merged = concatenate([x, y])
    # Fully connected layers for classification
    out = Dense(units=num_action_labels, activation='softmax')(merged)
    # Create the model
    model = Model(inputs=[input_pose, input_confidence], outputs=out)
    return model


# Initialize EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Loop over K-Folds
k_split_index = 0
for train_index, val_index in kf.split(x_train_pose, y_train_labels):
    print(f"k_split index: {k_split_index}")
    k_split_index += 1

    x_train_pose_fold, x_val_pose_fold = x_train_pose[train_index], x_train_pose[val_index]
    x_train_confidence_fold, x_val_confidence_fold = x_train_confidence[train_index], x_train_confidence[val_index]
    x_train_object_classes_fold, x_val_object_classes_fold = x_train_object_classes[train_index], x_train_object_classes[val_index]
    x_train_object_boxes_fold, x_val_object_boxes_fold = x_train_object_boxes[train_index], x_train_object_boxes[val_index]
    y_train_labels_fold, y_val_labels_fold = y_train_labels[train_index], y_train_labels[val_index]

    # Define and compile the model for each fold (you can use the same model you defined earlier)
    poseobj_model = create_model(input_shape_pose, input_shape_confidence, input_shape_object_classes, input_shape_object_boxes, num_action_labels)
    pose_model = create_base_model(input_shape_pose, input_shape_confidence, num_action_labels)
    poseobj_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    pose_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training Human pose and object model")
    # Train the model
    poseobj_model.fit([x_train_pose_fold, x_train_confidence_fold, x_train_object_classes_fold, x_train_object_boxes_fold],
              to_categorical(y_train_labels_fold),
              validation_data=([x_val_pose_fold, x_val_confidence_fold, x_val_object_classes_fold, x_val_object_boxes_fold],
                               to_categorical(y_val_labels_fold)),
              epochs=100, batch_size=32, callbacks=[early_stopping])

    print("Training human pose model")    
    # Train the model
    pose_model.fit([x_train_pose_fold, x_train_confidence_fold],
              to_categorical(y_train_labels_fold),
              validation_data=([x_val_pose_fold, x_val_confidence_fold],
                               to_categorical(y_val_labels_fold)),
              epochs=100, batch_size=32, callbacks=[early_stopping])
    
    print("calculating accuracy and loss")
    # Evaluate the model on the validation data
    poseobj_loss, poseobj_accuracy = poseobj_model.evaluate([x_val_pose_fold, x_val_confidence_fold, x_val_object_classes_fold, x_val_object_boxes_fold],
                                    to_categorical(y_val_labels_fold))
    
    pose_loss, pose_accuracy = pose_model.evaluate([x_val_pose_fold, x_val_confidence_fold],
                                    to_categorical(y_val_labels_fold))
    poseobj_accuracies.append(poseobj_accuracy)
    poseobj_losses.append(poseobj_loss)
    pose_accuracies.append(pose_accuracy)
    pose_losses.append(pose_loss)

    print("calculating performance metrics")
    # Print the final results
    poseobj_y_pred = poseobj_model.predict([x_val_pose_fold, x_val_confidence_fold, x_val_object_classes_fold, x_val_object_boxes_fold])
    pose_y_pred = pose_model.predict([x_val_pose_fold, x_val_confidence_fold])
    
    prec, recall, f1, mAP = calculate_performance_metrics(to_categorical(y_val_labels_fold), poseobj_y_pred)
    poseobj_precisions.append(prec)
    poseobj_recalls.append(recall)
    poseobj_f1s.append(f1)
    poseobj_maps.append(mAP)
    
    prec, recall, f1, mAP = calculate_performance_metrics(to_categorical(y_val_labels_fold), pose_y_pred)
    pose_precisions.append(prec)
    pose_recalls.append(recall)
    pose_f1s.append(f1)
    pose_maps.append(mAP)

    action_perc = calculate_action_performance(y_val_labels_fold, poseobj_y_pred)
    poseobj_label_percentages.append(action_perc)

    action_perc = calculate_action_performance(y_val_labels_fold, pose_y_pred)
    pose_label_percentages.append(action_perc)

    
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train the model
# # Make sure to prepare your data accordingly, including adjacency_matrix
# model.fit(
#     [x_train_pose, x_train_confidence, x_train_object_classes, x_train_object_boxes],
#     to_categorical(y_train_labels),
#     validation_data=([x_val_pose, x_val_confidence, x_val_object_classes, x_val_object_boxes], to_categorical(y_val_labels)),
#     epochs=20,
#     batch_size=32)

# # # with open("D:\\Users\\SMH\\Tim_v2\\data\\models\\cnn_model_v1.pkl", 'wb') as f:
# # #     pickle.dump(model, f)

# # save model
# model.save("D:\\Users\\SMH\\Tim_v2\\data\\models\\cnn_model_yolo_v1.h5")

# print("poseobj_accuracies: ", poseobj_accuracies)
# print("poseobj_losses: ", poseobj_losses)
# print("poseobj_precisions: ", poseobj_precisions)
# print("poseobj_recalls: ", poseobj_recalls)
# print("poseobj_f1s: ", poseobj_f1s)
# print("poseobj_maps: ", poseobj_maps)
# print("poseobj_label_percentages: ", poseobj_label_percentages)
# print("--------------------------------------------------")
# print("pose_accuracies: ", pose_accuracies)
# print("pose_losses: ", pose_losses)
# print("pose_precisions: ", pose_precisions)
# print("pose_recalls: ", pose_recalls)
# print("pose_f1s: ", pose_f1s)
# print("pose_maps: ", pose_maps)
# print("pose_label_percentages: ", pose_label_percentages)


    
print()
print("poseobj_accuracies =", round(np.mean(poseobj_accuracies)*100,2))
print("poseobj_losses =", round(np.mean(poseobj_losses),2))
print("poseobj_precisions =", round(np.mean(poseobj_precisions)*100,2))
print("poseobj_recalls =", round(np.mean(poseobj_recalls)*100,2))
print("poseobj_f1s =", round(np.mean(poseobj_f1s)*100,2))
print("poseobj_maps =", round(np.mean(poseobj_maps)*100,2))
print("poseobj_label_percentages: ", calculate_avg_label_percentage(poseobj_label_percentages))
print("--------------------------------------------------")
print("pose_accuracies =", round(np.mean(pose_accuracies)*100,2))
print("pose_losses =", round(np.mean(pose_losses),2))
print("pose_precisions =", round(np.mean(pose_precisions)*100,2))
print("pose_recalls =", round(np.mean(pose_recalls)*100,2))
print("pose_f1s =", round(np.mean(pose_f1s)*100,2))
print("pose_maps =", round(np.mean(pose_maps)*100,2))
print("pose_label_percentages: ", calculate_avg_label_percentage(pose_label_percentages))