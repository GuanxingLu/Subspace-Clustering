function [C,E,obj,err,iter] = tlrr_tnn_new(X,A,Omega,lambda,beta,s,opts)
% min_{C,E,B,Z,D} ||C||_{*}+lambda*||E||2_1+beta*trace(B*L*B')
% s.t. X=XZ+E, B=D, P(D)=P(A), Z=C(:,:,1), B=C(:,:,2).
%
% ---------------------------------------------
% Input:
%       X       -    d*n matrix
%       A       -    n*n matrix
%       Omega   -    index of the observed entries
%       lambda  -    >0, parameter
%       opts    -    Structure value in Matlab. The fields are
%           opts.tol        -   termination tolerance
%           opts.max_iter   -   maximum number of iterations
%           opts.mu         -   stepsize for dual variable updating in ADMM
%           opts.max_mu     -   maximum stepsize
%           opts.rho        -   rho>=1, ratio used to increase mu
%           opts.DEBUG      -   0 or 1
%
% Output:
%       C       -    n*n*2 tensor
%       E       -    d*n matrix
%       obj     -    objective function value
%       err     -    residual 
%       iter    -    number of iterations

tol = 1e-3; 
max_iter = 500;
rho = 1.1;
mu = 1e-3;
max_mu = 1e10;
DEBUG = 1;
if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'mu');          mu = opts.mu;                end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end
if isfield(opts, 'DEBUG');       DEBUG = opts.DEBUG;          end

X_size = size(X);   
n2 = X_size(2); % number of samples

% construct the Laplace matrix L = D - W;
Wpara.type='knn';
Wpara.k=1*floor(log2(n2)) + 1;  
[W,~]=genWv3(X',Wpara);
W(A>0) = 1; 
W(A<0) = 0;
DD=diag(sum(W,2));
L = DD-W;

Ck = zeros(n2,n2,2);
Y2 = Ck;
B = Ck(:,:,2);
Z = B;
D = B;
Y3 = D;
sA = s*A(Omega);
D(Omega) = sA;  
E = zeros(X_size);
Y1 = E;

XtX = X'*X;
I = eye(n2);
invXtXI = (I+XtX)\I;
%L = L+0.01*I;
Beta2L = beta*(L+L');
   
for iter = 1 : max_iter
    Bk = B;
    Zk = Z;
    Ek = E;
    Dk = D;
    Ck(:,:,2) = B;  
    Ck(:,:,1) = Z;
    
    % update C
    [C,tnnC,~] = prox_tnn(Ck + Y2/mu,1/mu);
    
    % update Z
    Z = invXtXI * (XtX + X'*(-E+Y1/mu) + C(:,:,1) - Y2(:,:,1)/mu);

    % update B
    B = (mu*(C(:,:,2)+D) - (Y2(:,:,2)+Y3))/(Beta2L + 2*mu*I);
    
    % update D
    D = B + Y3/mu;
    D(Omega) = sA;   

    % update E
    E = prox_l21(X - X*Z + Y1/mu,lambda/mu);    %l21
    
    dY1 = X - X*Z - E;
    dY2 = Ck - C;
    dY3 = B - D;
    chgC = max(abs(Ck(:)-C(:)));
    chgB = max(abs(Bk(:)-B(:)));
    chgZ = max(abs(Zk(:)-Z(:)));
    chgE = max(abs(Ek(:)-E(:)));
    chgD = max(abs(Dk(:)-D(:)));
    chg = max([ chgC chgB chgZ chgE chgD max(abs(dY1(:))) max(abs(dY2(:))) max(abs(dY3(:)))]);
    
    %ERR(iter) = chg;
    if DEBUG
        if iter == 1 || mod(iter, 10) == 0
            obj = tnnC + lambda*norm21(E(:))+beta*trace(B*L*B'); %l21
            err = norm(dY1(:)); 
             disp(['iter ' num2str(iter) ', mu=' num2str(mu) ...
                    ', obj=' num2str(obj) ', err=' num2str(err) ...
                    ', chg=' num2str(chg)]); 
        end
    end
    if chg < tol
        break;
    end 
    Y1 = Y1 + mu*dY1;
    Y2 = Y2 + mu*dY2;
    Y3 = Y3 + mu*dY3;
    mu = min(rho*mu,max_mu);   
end
obj = tnnC+lambda*norm21(E(:))+beta*trace(B*L*B');     %l21
err = norm(dY1(:));

