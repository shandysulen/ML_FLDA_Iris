from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, inv
import pandas as pd
from sklearn.datasets import load_iris

def create_scatterplot(ax, x1, y1, label1, x2, y2, label2, x_label, y_label, title):
    """ Utility function to create a 2D scatterplot with matplotlib.pyplot """
    ax.scatter(x1, y1, c='b', marker='o', label=label1)
    ax.scatter(x2, y2, c='r', marker='x', label=label2)
    ax.legend(loc='best', fancybox=True, shadow=True, fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

def find_threshold(patterns, labels):
    min_misclass_errors = len(patterns)
    best_threshold = None

    for i in patterns:
        pos = i + 0.000000001
        misclass_errors = 0
        for val_idx in range(len(patterns)):
            if patterns[val_idx] < pos and labels[val_idx] == 2:
                misclass_errors += 1
            if patterns[val_idx] > pos and labels[val_idx] == 1:
                misclass_errors += 1
        if misclass_errors < min_misclass_errors:
            min_misclass_errors = misclass_errors
            best_threshold = pos

    return (best_threshold, min_misclass_errors)

# Load Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target
label_names = iris.target_names
features = iris.feature_names

# Plot preprocessing
setosa_sepal_length = X[:50,0]
setosa_sepal_width = X[:50,1]
setosa_petal_length = X[:50,2]
setosa_petal_width = X[:50,3]
versicolor_sepal_length = X[50:100,0]
versicolor_sepal_width = X[50:100,1]
versicolor_petal_length = X[50:100,2]
versicolor_petal_width = X[50:100,3]
virginica_sepal_length = X[100:150,0]
virginica_sepal_width = X[100:150,1]
virginica_petal_length = X[100:150,2]
virginica_petal_width = X[100:150,3]

# 2D scatterplots showing feature relationships between classes
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), num='Setosa & Versicolor Scatterplots')
plt.subplots_adjust(hspace=0.52)
create_scatterplot(axes[0][0], setosa_sepal_length, setosa_sepal_width, 'Setosa',
                   versicolor_sepal_length, versicolor_sepal_width, 'Versicolor',
                   'Sepal Length (cm)', 'Sepal Width (cm)', 
                   'Sepal Width (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[0][1], setosa_sepal_length, setosa_petal_length, 'Setosa',
                   versicolor_sepal_length, versicolor_petal_length, 'Versicolor',
                   'Sepal Length (cm)', 'Petal Length (cm)', 
                   'Petal Length (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[1][0], setosa_sepal_length, setosa_petal_width, 'Setosa',
                   versicolor_sepal_length, versicolor_petal_width, 'Versicolor',
                   'Sepal Length (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[1][1], setosa_sepal_width, setosa_petal_length, 'Setosa',
                   versicolor_sepal_width, versicolor_petal_length, 'Versicolor',
                   'Sepal Width (cm)', 'Petal Length (cm)', 
                   'Petal Length (cm) vs. Sepal Width (cm)')
create_scatterplot(axes[2][0], setosa_sepal_width, setosa_petal_width, 'Setosa',
                   versicolor_sepal_width, versicolor_petal_width, 'Versicolor',
                   'Sepal Width (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Sepal Width (cm)')
create_scatterplot(axes[2][1], setosa_petal_length, setosa_petal_length, 'Setosa',
                   versicolor_petal_length, versicolor_petal_length, 'Versicolor',
                   'Petal Length (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Petal Length (cm)')
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), num='Setosa & Virginica Scatterplots')
plt.subplots_adjust(hspace=0.52)
create_scatterplot(axes[0][0], setosa_sepal_length, setosa_sepal_width, 'Setosa',
                   virginica_sepal_length, virginica_sepal_width, 'Virginica',
                   'Sepal Length (cm)', 'Sepal Width (cm)', 
                   'Sepal Width (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[0][1], setosa_sepal_length, setosa_petal_length, 'Setosa',
                   virginica_sepal_length, virginica_petal_length, 'Virginica',
                   'Sepal Length (cm)', 'Petal Length (cm)', 
                   'Petal Length (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[1][0], setosa_sepal_length, setosa_petal_width, 'Setosa',
                   virginica_sepal_length, virginica_petal_width, 'Virginica',
                   'Sepal Length (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[1][1], setosa_sepal_width, setosa_petal_length, 'Setosa',
                   virginica_sepal_width, virginica_petal_length, 'Virginica',
                   'Sepal Width (cm)', 'Petal Length (cm)', 
                   'Petal Length (cm) vs. Sepal Width (cm)')
create_scatterplot(axes[2][0], setosa_sepal_width, setosa_petal_width, 'Setosa',
                   virginica_sepal_width, virginica_petal_width, 'Virginica',
                   'Sepal Width (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Sepal Width (cm)')
create_scatterplot(axes[2][1], setosa_petal_length, setosa_petal_length, 'Setosa',
                   virginica_petal_length, virginica_petal_length, 'Virginica',
                   'Petal Length (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Petal Length (cm)')
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12), num='Versicolor & Virginica Scatterplots')
plt.subplots_adjust(hspace=0.52)
create_scatterplot(axes[0][0], versicolor_sepal_length, versicolor_sepal_width, 'Versicolor',
                   virginica_sepal_length, virginica_sepal_width, 'Virginica',
                   'Sepal Length (cm)', 'Sepal Width (cm)', 
                   'Sepal Width (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[0][1], versicolor_sepal_length, versicolor_petal_length, 'Versicolor',
                   virginica_sepal_length, virginica_petal_length, 'Virginica',
                   'Sepal Length (cm)', 'Petal Length (cm)', 
                   'Petal Length (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[1][0], versicolor_sepal_length, versicolor_petal_width, 'Versicolor',
                   virginica_sepal_length, virginica_petal_width, 'Virginica',
                   'Sepal Length (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Sepal Length (cm)')
create_scatterplot(axes[1][1], versicolor_sepal_width, versicolor_petal_length, 'Versicolor',
                   virginica_sepal_width, virginica_petal_length, 'Virginica',
                   'Sepal Width (cm)', 'Petal Length (cm)', 
                   'Petal Length (cm) vs. Sepal Width (cm)')
create_scatterplot(axes[2][0], versicolor_sepal_width, versicolor_petal_width, 'Versicolor',
                   virginica_sepal_width, virginica_petal_width, 'Virginica',
                   'Sepal Width (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Sepal Width (cm)')
create_scatterplot(axes[2][1], versicolor_petal_length, versicolor_petal_length, 'Versicolor',
                   virginica_petal_length, virginica_petal_length, 'Virginica',
                   'Petal Length (cm)', 'Petal Width (cm)', 
                   'Petal Width (cm) vs. Petal Length (cm)')
plt.show()

# Compute and print mean vector for each class
u = {'setosa': np.zeros(4), 'virginica': np.zeros(4), 'versicolor': np.zeros(4)}

for i in range(4):
    u['setosa'][i] = sum(X[0:50,i]) / 50
    u['virginica'][i] = sum(X[50:100, i]) / 50
    u['versicolor'][i] = sum(X[100:150, i]) / 50

for key, value in u.items():
    print(f'{key} class mean vector: {value}')

# Compute within-class scatter matrices (Sw_set_vers, Sw_set_virg, Sw_vers_virg)
Sw_set_vers = np.array(np.zeros((4,4)))
for x in X[0:50, :]:
    Sw_set_vers += np.outer(x - u['setosa'], x - u['setosa'])
for x in X[50:100, :]:
    Sw_set_vers += np.outer(x - u['versicolor'], x - u['versicolor'])

Sw_set_virg = np.array(np.zeros((4,4)))
for x in X[0:50, :]:
    Sw_set_virg += np.outer(x - u['setosa'], x - u['setosa'])
for x in X[100:150, :]:
    Sw_set_virg += np.outer(x - u['virginica'], x - u['virginica'])  

Sw_vers_virg = np.array(np.zeros((4,4)))
for x in X[0:50, :]:
    Sw_vers_virg += np.outer(x - u['versicolor'], x - u['versicolor'])
for x in X[100:150, :]:
    Sw_vers_virg += np.outer(x - u['virginica'], x - u['virginica'])

print(f'\nwithin-class scatter matrix (setosa, versicolor):\n{Sw_set_vers}')
print(f'\nwithin-class scatter matrix (setosa, virginica):\n{Sw_set_virg}')
print(f'\nwithin-class scatter matrix (versicolor, virginica):\n{Sw_vers_virg}')

# Compute between-class scatter matrix (Sb_set_vers, Sb_set_virg, Sb_vers_virg)
Sb_set_vers = np.outer(u['versicolor'] - u['setosa'], u['versicolor'] - u['setosa'])
Sb_set_virg = np.outer(u['virginica'] - u['setosa'], u['virginica'] - u['setosa'])
Sb_vers_virg = np.outer(u['virginica'] - u['versicolor'], u['virginica'] - u['versicolor'])

print(f'\nbetween-class scatter matrix (setosa, versicolor):\n{Sb_set_vers}')
print(f'\nbetween-class scatter matrix (setosa, virginica):\n{Sw_set_virg}')
print(f'\nbetween-class scatter matrix (versicolor, virginica):\n{Sw_vers_virg}')

# Solve generalized eigenvalue problem for eigenvector/eigenvalue pairs
eig_vals, eig_vecs = eig(np.dot(inv(Sw_set_vers), Sb_set_vers))
sorted_eig_pairs_set_vers = sorted(zip(eig_vals, eig_vecs), reverse=True)
eig_vals, eig_vecs = eig(np.dot(inv(Sw_set_virg), Sb_set_virg))
sorted_eig_pairs_set_virg = sorted(zip(eig_vals, eig_vecs), reverse=True)
eig_vals, eig_vecs = eig(np.dot(inv(Sw_vers_virg), Sb_vers_virg))
sorted_eig_pairs_vers_virg = sorted(zip(eig_vals, eig_vecs), reverse=True)

# Choose eigenvector pair with greatest eigenvalue
w_set_vers = sorted_eig_pairs_set_vers[0][1]
print(f"weight vector (Setosa, Versicolor): {w_set_vers}\n")
w_set_virg = sorted_eig_pairs_set_virg[0][1]
print(f"weight vector (Setosa, Virginica): {w_set_virg}\n")
w_vers_virg = sorted_eig_pairs_vers_virg[0][1]
print(f"weight vector (Versicolor, Virginica): {w_vers_virg}\n")

# Project 4D data into a 1-dimensional subspace
y_set_vers = np.dot(X[:100,:], w_set_vers)
y_set_virg = np.dot(np.vstack((X[0:50,:],X[100:150,:])), w_set_virg)
y_vers_virg = np.dot(X[50:150,:], w_vers_virg)

# Show 1D scatterplots
fig, axes = plt.subplots(nrows=1, ncols=3, clear=True, num='FLDA 1D Scatterplots (Histograms)', figsize=(14,8))

axes[0].hist(y_set_vers[:50], color="#FF000088", label='Setosa', rwidth=0.90, align='left', bins=15)
axes[0].hist(y_set_vers[50:100], color="#00FF0088", label='Versicolor', rwidth=0.90, align='left', bins=15)
axes[0].tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=True
)
axes[0].set_ylabel('count')
axes[0].legend()
axes[0].set_title('Setosa vs. Versicolor 1D Scatterplot (Histogram)')

axes[1].hist(y_set_virg[:50], color="#FF000088", label='Setosa', rwidth=0.90, align='left', bins=15)
axes[1].hist(y_set_virg[50:], color="#0000FF88", label='Virginica', rwidth=0.90, align='left', bins=15)
axes[1].tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=True
)
axes[1].set_ylabel('count')
axes[1].legend()
axes[1].set_title('Setosa vs. Virginica 1D Scatterplot (Histogram)')

axes[2].hist(y_vers_virg[:50], color="#00FF0088", label='Versicolor', rwidth=0.90, align='left', bins=15)
axes[2].hist(y_vers_virg[50:], color="#0000FF88", label='Virginica', rwidth=0.90, align='left', bins=15)
axes[2].tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=True
)
axes[2].set_ylabel('count')
axes[2].legend()
axes[2].set_title('Versicolor vs. Virginica 1D Scatterplot (Histogram)')

plt.subplots_adjust(left=0.05, right=0.95)
plt.show()

# Setosa & Versicolor Histogram with Threshold Value 1.65
fig, ax = plt.subplots(clear=True, num='Setosa vs. Versicolor Histogram (Threshold Value = 1.65)')
ax.hist(y_set_vers[:50], color="#FF000088", label='Setosa', rwidth=0.90, align='left', bins=15)
ax.hist(y_set_vers[50:100], color="#00FF0088", label='Versicolor', rwidth=0.90, align='left', bins=15)
ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=True
)
ax.set_ylabel('count')
plt.axvline(x=1.65, color='black', label='Threshold Value = 1.65')
ax.legend()
ax.set_title('Setosa vs. Versicolor Histogram (Threshold Value = 1.65)\n Misclassification Errors: 0')
plt.show()

# Setosa & Virginica Histogram with Threshold Value -2.98
fig, ax = plt.subplots(clear=True, num='Setosa vs. Virginica Histogram (Threshold Value = -2.98)')
ax.hist(y_set_virg[:50], color="#FF000088", label='Setosa', rwidth=0.90, align='left', bins=15)
ax.hist(y_set_virg[50:100], color="#0000FF88", label='Virginica', rwidth=0.90, align='left', bins=15)
ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=True
)
ax.set_ylabel('count')
plt.axvline(x=-2.976, color='black', label='Threshold Value = -2.98')
ax.legend()
ax.set_title('Setosa vs. Virginica Histogram (Threshold Value = -2.98)\n Misclassification Errors: 0')
plt.show()

# Find threshold for Versicolor and Virginica
(threshold, misclass_errors) = find_threshold(y_vers_virg, y[50:])

# Versicolor & Virginica Histogram with Threshold Value 1.7364729424
fig, ax = plt.subplots(clear=True, num=f'Versicolor vs. Virginica Histogram (Threshold Value = {threshold})')
ax.hist(y_vers_virg[:50], color="#00FF0088", label='Versicolor', rwidth=0.10, align='left', bins=100)
ax.hist(y_vers_virg[50:100], color="#0000FF88", label='Virginica', rwidth=0.10, align='left', bins=100)
ax.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=True
)
ax.set_ylabel('count')
plt.axvline(x=1.74, color='black', label=f'Threshold Value = {threshold}', alpha=0.4)
ax.legend()
ax.set_title(f'Versicolor vs. Virginica Histogram (Threshold Value = {threshold})\n Misclassification Errors: {misclass_errors}')
plt.show()