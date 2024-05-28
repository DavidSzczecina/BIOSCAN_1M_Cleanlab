#!pip install matplotlib torch torchvision skorch
#!pip install cleanlab

import numpy as np
import torch
import warnings
from sklearn.datasets import fetch_openml
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from cleanlab.filter import find_label_issues


#Set a seed for reproducibility
def setSeed():
    SEED = 321 #any constant value
    np.random.seed(SEED)  #using the same seed for numpy and torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(SEED)
    warnings.filterwarnings("ignore", "Lazy modules are a new feature.*") #ignore warning related to lazy modules





#Define a classification model
#currently using baisc model, needs to swap to resnet50
def setModel():
    class ClassifierModule(nn.Module):
        def __init__(self):
            super().__init__()

            self.cnn = nn.Sequential(
                nn.Conv2d(1, 6, 3),
                nn.ReLU(),
                nn.BatchNorm2d(6),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, 3),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.out = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(dim=-1),
            )

        def forward(self, X):
            X = self.cnn(X)
            X = self.out(X)
            return X
    model = ClassifierModule()
    return model  # Return the created ClassifierModule instance

  #Compute out-of-sample predicted probabilities
    #pass the correct arguments
def outOfSamplePredProbs(ClassifierModule, datasets, labels):
    model_skorch = NeuralNetClassifier(ClassifierModule)
    num_crossval_folds = 3  # for efficiency; values like 5 or 10 will generally work better
    pred_probs = cross_val_predict(
        model_skorch,
        dataset,
        labels,
        cv=num_crossval_folds,
        method="predict_proba",
    )
    predicted_labels = pred_probs.argmax(axis=1)
    acc = accuracy_score(labels, predicted_labels)
    return pred_probs, predicted_labels, acc


#find the label issues
def findLabelIssues(labels ,pred_probs):
    ranked_label_issues = find_label_issues(
    labels,
    pred_probs,
    return_indices_ranked_by="self_confidence",
    frac_noise=1
    )
    return ranked_label_issues

    #print(f"Cleanlab found {len(ranked_label_issues)} label issues.")
    #print(f"Top 15 most likely label errors: \n {ranked_label_issues[:15]}")



def main() -> None:
    setSeed()
    model = setModel()

    data_train = dataset_train.data
    data_labels = new_labels

    pred_probs, predicted_labels, acc = outOfSamplePredProbs(model, data_train, data_labels)
    label_issues = findLabelIssues(new_labels, pred_probs)


    #here are the labels that are most likely corrupted
    print(f"Label issues: {label_issues}")

    mislabel_indices = label_issues
    mislabel_indices.sort(reverse=True)
    pruned_dataset = np.delete(data_train, mislabel_indices, axis=0)
    pruned_labels = np.delete(data_labels, mislabel_indices, axis=0)


if __name__ == '__main__':
    main()
