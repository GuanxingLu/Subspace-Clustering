function [W,L]=genWv3(X,Wpara)
% constructed W based on GSP toolbox.
% Graph regularization is used to obtain smooth C.

% X=X';

param.type=Wpara.type;
param.k=Wpara.k;
% param.sigma=Wpara.sigma2;
% param.use_full=1;
% param.rescale=1;
% param.epsilon=Wpara.epsilon;

G = gsp_nn_graph(X, param);
W=full(G.W);
L=full(G.L);

% G.W(:,1)

%% some important parameters for gsp_nn_graph
% param.rescale : [0, 1] rescale the data on a 1-ball (def 0)
% param.k : int number of neighbors for knn (def 10)
% param.epsilon : float the radius for the range search
% param.sigma : float the variance of the distance kernel
% param.type : ['knn', 'radius'] the type of graph (default 'knn')
% param.use_full : [0, 1] - Compute the full distance matrix and then sparsify it (default 0)
% param.epsilon : float the radius for the range search
