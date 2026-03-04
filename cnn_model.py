# imports


import pickle
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf




from sklearn.metrics import (accuracy_score, precision_score, recall_score, confusion_matrix)
from tensorflow.keras.models import Sequential

# Import nn layers

from tensorflow.keras.layers import (
    Conv1D,                     
    MaxPooling1D,               # Downsamples feature maps
    GlobalAveragePooling1D, 
    Input,                      
    Dense,                      # Fully connected layer
    Dropout,                    # Prevents overfitting
    BatchNormalization,        
                         
)


np.random.seed(42)                  # so that i would get same random numbers every time i run the code
tf.random.set_seed(42)              # for reproducibility in TensorFlow
random.seed(42)                     # consistent random behavior everytime i run it



with open("Dataset/breathing_dataset.pkl", "rb") as f:
    data = pickle.load(f)





X = data["X"]
y = data["y"]
participants = data["participants"]

print("X shape:", X.shape)

unique_labels, counts = np.unique(y, return_counts=True)

print("Original Label Distribution:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count}")







# Convert to binary labels: Normal vs Abnormal, why i did this?
#  because i want to -
#  focus on distinguishing normal breathing from any kind of abnormality,
#  which is more relevant and also helps with class imbalance issues.
#  By grouping all abnormalities together, we can train a model that is better at detecting any deviation from normal breathing patterns
#  rather than trying to differentiate between specific types of abnormalities which may have limited data.


y_binary = np.array(
    ["Normal" if label == "Normal" else "Abnormal" for label in y]
)

le = LabelEncoder()
y_encoded = le.fit_transform(y_binary)           # Abnormal=0, Normal=1
classes = le.classes_

print("Binary Label Distribution:",
      dict(zip(*np.unique(y_binary, return_counts=True))))









def build_model(input_shape):
    model = Sequential([

                                                
        Input(shape=input_shape),                                   # 30-sec window - time steps × channels)
        Conv1D(16, 7, activation="relu", padding="same"),           # the model can learn 16 different types of patterns, kernel size 7 times steps the filter looks at at once.
        BatchNormalization(),
        MaxPooling1D(4),                                            # reduces the size of the signal while keeping the most important information creates pair and choose the max values from them if 2 but here i took 4 

        
        # again

        Conv1D(32, 5, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(4),


        # Third Conv block
        Conv1D(64, 3, activation="relu", padding="same"),
        BatchNormalization(),
        GlobalAveragePooling1D(),

        Dense(32, activation="relu"),                                # Fully connected layer
        Dense(2, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",    # used when labels are integers like 0,1
        metrics=["accuracy"]
    )

    return model








unique_participants = np.unique(participants)                   # Get unique participant IDs
results_list = []
all_true = []
all_pred = []   

for test_p in unique_participants:
    train = participants != test_p
    test = participants == test_p



    X_train, X_test = X[train], X[test]                         # test train split based on participant
    y_train, y_test = y_encoded[train], y_encoded[test]



    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8        # keepdims for clean broadcasting
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    model = build_model(X_train.shape[1:])



    class_weights = {0: 4.0, 1: 1.0}                            # Giving more weight to the minority class (Abnormal) to help the model learn better from it



    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        class_weight=class_weights,                             # as it's a imbalanced dataset
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0)
    abnormal_prob = y_prob[:, 0]

    y_pred = np.where(abnormal_prob > 0.5, 0, 1)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")


    results_list.append({
        "Participant": test_p,
        "Fold_Accuracy": round(acc, 4)
    })

    all_true.extend(y_test)
    all_pred.extend(y_pred)





























# final report







all_true = np.array(all_true)
all_pred = np.array(all_pred)

overall_accuracy = accuracy_score(all_true, all_pred)
overall_precision = precision_score(all_true, all_pred, average="weighted", zero_division=0)
overall_recall = recall_score(all_true, all_pred, average="weighted", zero_division=0)
overall_cm = confusion_matrix(all_true, all_pred)


print("final report LOPO")

print(f"Accuracy  : {overall_accuracy:.4f}")
print(f"Precision : {overall_precision:.4f}")
print(f"Recall    : {overall_recall:.4f}")
print("\nConfusion Matrix:")
print(pd.DataFrame(overall_cm,
                   index=classes,
                   columns=classes))





# save confusion matrix as heatmap


fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(pd.DataFrame(overall_cm,
                         index=classes,
                         columns=classes),
            annot=True, fmt="d", cmap="Blues", ax=ax)

ax.set_title("Overall Confusion Matrix")
ax.set_ylabel("True Label")
ax.set_xlabel("Predicted Label")

plt.tight_layout()
plt.savefig("Results/confusion_matrix.png", dpi=120)
plt.close()



#  save csv


fold_df = pd.DataFrame(results_list)

overall_row = pd.DataFrame([{
    "Participant": "OVERALL",
    "Fold_Accuracy": round(overall_accuracy, 4),
    "Final_Precision": round(overall_precision, 4),
    "Final_Recall": round(overall_recall, 4),
    "TP_Abnormal": overall_cm[0, 0],
    "FN_Abnormal": overall_cm[0, 1],
    "FP_Normal": overall_cm[1, 0],
    "TN_Normal": overall_cm[1, 1]
}])

final_df = pd.concat([fold_df, overall_row], ignore_index=True)

final_df.to_csv("Results/final_results.csv", index=False)

print("\nSaved Files:")
print(" - Results/confusion_matrix.png")
print(" - Results/final_results.csv")