%--------------------------------------------------------------------------
% TO DO - notation description here - for now see paper!
%--------------------------------------------------------------------------
% Krishna Somandepalli, Nov 2017
% Paper citation
% K. Somandepalli and S. Narayanan, "Reinforcing Self-expressive Representation with Constraint Propagation for Face Clustering in Movies," ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 4065-4069.
%--------------------------------------------------------------------------
% e.g. [C,WF, err1J, err2J, err3J] = cp_ssc(X, Y_incomplete, false, l);
%             F_joint = WF(size(X,1)+1:end, :);
%             C_joint = C(1:N, :);

function [C2, B, err_1, err_2, err_3] = cp_ssc(X, Y, affine, l)%,alpha,thr,maxIter)
% Y = A %-1,0,1

tol = 1e-3;

% affine = false;
maxiter = 1000;
%num of samples = n; feature dim=d
[d,n] = size(X);
m = d+n;

% find some params
mu_z1 = computeLambda_mat(X);
% mu_z2 = computeLambda_mat([X;Y]);

%mu_z2 = lansvd(sign([X;Y]), 1, 'L');
%mu_z2 = lansvd(sign([X;Y]));
mu_z2 = norm(sign([X;Y]), 2);

%mu_z1 = 1e-3;
%mu_z2 = 1e-3;

alpha = 10;

alpha1 = alpha/mu_z1;
alpha2 = mu_z2;

% set all mu
mu1 = alpha1;
mu2 = alpha;
mu3 = 0.1/alpha2;

% little lambda: l1 weighs ||C||_1, l2 weighs ||(X;Y)||_*
% pick a lil lambda such that the weights on l1 and l* norm sum to 1 
% l = 1000;
l1 = (l/(1+l));
l2 = (1/(1+l));

% lagrange multipliers that go into trace
L2 = zeros(n);
L3 = zeros(m,n);

% init
C = zeros(n); % SSC
B = zeros(m,n); % MC
% need to update only the lower nxn!
M = zeros(m,n); % MC

% all entries that belong to cannot link pairs = function psi
C_cannot = (Y<=-1);

% function for soft thresholding - inputs matrix, threshold, size of the
% matrix
soft_threshold = @(K,beta,n)(max(0, abs(K) - (beta*ones(n)) ) .* sign(K));

% inf norm error between vectorized matrices
inf_norm_err = @(K1, K2)(max(max( abs(K1-K2) )));

% update_param = []
err_1 = 10*tol; err_2 = 10*tol; err_3 = 10*tol; err_4 = 10*tol;

iter = 1;
if (~affine)
    
    G = X'*X;
    H = inv( mu1*G + mu2*eye(n) );
    
    while (err_1(iter) > tol && err_3(iter) > tol && iter < maxiter)
        %update J
        J = H * ((mu1*G) + (mu2*C) - L2);
        J = J - (C_cannot .* J);
        J = J - diag(diag(J)) ;

        % update C - soft thresholding
        C2 = soft_threshold( J + (L2/mu2), l1/mu2, n);
%       C2 = max(0,(abs(J+L2/mu2) - (l1/mu2)*ones(n))) .* sign(J+(L2/mu2));
        C2 = C2 - (C_cannot .* C2);
        C2 = C2 - diag(diag(C2));

        C = C2;
        
%         update_param = [update_param, ]        
        % create data matrix for matrix completion
        if iter==1
            N = [X ; Y];
        else
            N = [X*J ; Y];
        end
        
        % update R
        % R = Y - F + L3/mu3
        M(d+1:end, :) = Y - B(d+1:end,:) + (L3(d+1:end,:)/mu3);
        % function phi
        M(N~=0) = 0;
               
        % update B  -singular value soft thresholding
        [U,S,V] = svd(N - M + (L3/mu3), 'econ');
        % note that singular values are always positive!
        % and singular values come sorted in matlab
        diag_S = diag(S);
        beta_2 = l2/mu3;
        svp = length( find(diag_S >  beta_2) );
        B = U(:,1:svp) * diag( diag_S(1:svp) - beta_2 ) * V(:,1:svp)';
        
        N_err = N - M - B;
        
        % update C_cannot
        C_cannot = (B(d+1:end,:)<=-1);
        
        % update big lambda
        L2 = L2 + mu2 * (J - C2);
        L3 = L3 + mu3 * N_err; 
        
        % update mu3
        rho_s = length(N~=0)/(m*n);
        rho = 1.1 + 2.5*rho_s;
        mu3 = rho*mu3;
        
        
        % find errors in the SSC part
        err_1(iter+1) = inf_norm_err(J,C2);
        err_2(iter+1) = errorLinSys(X, J);

        % find errors in the MC part
        err_3(iter+1) = norm(N_err, 'fro')/norm(N, 'fro');
        %if mod(iter, 10)==0
            %disp('iter=' num2striter, err_1(end), err_2(end), err_4(end), err_3(end), svp]);
            disp(['iter ' num2str(iter) ', err1=' num2str(err_1(end)) ...
            ', err2=' num2str(err_2(end)) ', err3=' num2str(err_3(end))]); 
        %end
        iter = iter + 1;
        
%         subplot(121); imagesc(abs(C2)+abs(C2')); subplot(122); imagesc(B(d+1:end,:))
%         pause(0.1)
        
    end
    
else
    G = X'*X;
    H = inv( mu1*G + mu2*eye(n) + 1*mu2*ones(n));
    L4 = ones(1,n);
    
    while ((err_1(iter) > tol || err_4(iter) > tol) && err_3(iter) > tol...
            && iter < maxiter)
        %update J
        J = H * ((mu1*G) + (1*mu2*C) - (1*L2) + (mu2*ones(n)) - (ones(n,1)*L4));
        J = J - (C_cannot .* J);
        J = J - diag(diag(J)) ;

        % update C - soft thresholding
        C2 = soft_threshold( J + (L2/mu2), l1/(1*mu2), n);
%         C2 = max(0,(abs(J+L2/mu2) - (l1/mu2)*ones(n))) .* sign(J+(L2/mu2));
        
        C2 = C2 - (C_cannot .* C2);
        C2 = C2 - diag(diag(C2));
        C = C2;

%         update_param = [update_param, ]
        
        % create data matrix for matrix completion
        N = [X*J ; Y];
%         imagesc(C_cannot);
%         pause(0.1)
        % update R
        % R = Y - F + L3/mu3
        M(d+1:end, :) = Y - B(d+1:end,:) + (L3(d+1:end,:)/mu3);
        % function phi
        M(N~=0) = 0;
               
        % update B  -singular value soft thresholding
        [U,S,V] = svd(N - M + (L3/mu3), 'econ');
        % note that singular values are always positive!
        % and singular values come sorted in matlab
        diag_S = diag(S);
        beta_2 = l2/mu3;
        svp = length( find(diag_S >  beta_2) );
        B = U(:,1:svp) * diag( diag_S(1:svp) - beta_2 ) * V(:,1:svp)';
        
        N_err = N - M - B;
        
        % update C_cannot
        C_cannot = (B(d+1:end,:)<=-1);
        
        % update big lambda
        L2 = L2 + mu2 * (J - C2);
        L3 = L3 + mu3 * N_err; 
        L4 = L4 + mu2 * (ones(1,n)*J - ones(1,n));
        % update mu3
        rho_s = length(N~=0)/(m*n);
        rho = 1.1 + rho_s;
        mu3 = rho*mu3;

        
        
        % find errors in the SSC part
        err_1(iter+1) = inf_norm_err(J,C2);
        err_2(iter+1) = errorLinSys(X, J);
        err_4(iter+1) = inf_norm_err(ones(1,n)*J,ones(1,n));
        
        % find errors in the MC part
        err_3(iter+1) = norm(N_err, 'fro')/norm(N, 'fro');
        
        %if mod(iter, 10)==0
            %disp('iter=' num2striter, err_1(end), err_2(end), err_4(end), err_3(end), svp]);
            disp(['iter ' num2str(iter) ', err1=' num2str(err_1(end)) ...
            ', err2=' num2str(err_2(end)) ', err3=' num2str(err_3(end)) ...
            ', err4=' num2str(err_4(end))]); 
        %end
        iter = iter + 1;
        
        
%         subplot(121); imagesc(abs(C2)+abs(C2')); subplot(122); imagesc(B(d+1:end,:))
%         pause(0.1)
    end
    
end

end
