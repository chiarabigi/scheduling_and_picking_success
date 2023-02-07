# SCHEDULING AND PICKING SUCCESS
using GNN 

## General idea

Make every strawberry learn it's representation in the contest of the cluster.

Clusters are represented as graphs: each node has, as features, bbox coordinates and occlusion propery as a weight. All nodes are connected to each other, the edge feature is the pixel euclidean distance.

The models use GATConv layers, which are like convolutional layers but for graphs. They are used so every nodes recieves informations from its neighbours, and aggregates them with their own. The attention in this layer is used because not all edges have the same importance.

The scheduling model is a node classification model. It classifies nodes as "first strawberry to be picked" or not.

The picking success model has, in the dataset, another node feature: isTarget. It is a graph classification problem, the output is a probabilty of the target strawberry being succesfully picked.

The usage of the two models together is explained in the "experiments.py" file
