%% The implementation of our paper
% Semi-Supervised Subspace Clustering via Tensor Low-Rank Representation
% https://arxiv.org/abs/2205.10481
% before running the code, you need download libraries from 
% https://github.com/GuanxingLu/Subspace-Clustering

clc,clear
addpath(genpath(cd))
delete(gcp('nocreate'));
parpool('local',12);

testset_p = 0.05:0.05:0.3;
acc_p_proposed = zeros(2,length(testset_p));
acc_p_lrr = acc_p_proposed;
acc_p_dplrr = acc_p_proposed;
acc_p_sslrr = acc_p_proposed;
acc_p_lrpca = acc_p_proposed;
acc_p_cpssc = acc_p_proposed;
acc_p_sclrr = acc_p_proposed;
acc_p_clrr = acc_p_proposed;
nmi_p_proposed = zeros(2,length(testset_p));
nmi_p_lrr = acc_p_proposed;
nmi_p_dplrr = acc_p_proposed;
nmi_p_sslrr = acc_p_proposed;
nmi_p_lrpca = acc_p_proposed;
nmi_p_cpssc = acc_p_proposed;
nmi_p_sclrr = acc_p_proposed;
nmi_p_clrr = acc_p_proposed;

f = {'ORL400_new';'YaleB944';'COIL20';'Isolet1';'2k2k_new';...
    'Alphabet';'BF0502';'Notting-Hill'};
f1 = {'ORL';'YaleB';'COIL20';'Isolet';'MNIST';...
    'Alphabet';'BF0502';'Notting-Hill'};
%% Parameter settings
%Proposed
lambda_tlrr_fileset = [5 1 0.5 0.5 0.5 0.1 0.1 0.01];
beta_tlrr_fileset   = [0 0 10  10 10 10 10 10];
%DPLRR
alpha_dplrr_fileset = [10 10   0.1  10   1    1    1  0.1];
beta_dplrr_fileset =  [1  0.01 0.01 0.01 0.01 0.01 10 0.01];
%LRR
lambda_lrr_fileset = [5 1 0.1 0.5 0.1 0.05 0.1 0.05];
opts.DEBUG = 1;
opts.tol = 1e-3;
%SSLRR
lambda_sslrr_fileset = [5 1 0.5 0.5 0.5 1 5 1];
%L-RPCA
w = 1;
lambda_lrpca_fileset = [0.1 1 0.05 0.5 1 0.05 0.01 0.01];
%CP-SSC
lambda_cpssc_fileset = [0.05 0.01 10 0.01 0.01 10 5 0.05];
%SC-LRR
lambda_sclrr_fileset = [10 5 0.5 0.1 1 1 1 5];
beta_sclrr_fileset =   [1  1 1   0.1 1 1 1 1]; 
%CLRR
lambda_clrr_fileset = [5 1 1 0.5 0.5 0.05 0.5 1];

tic
%% file loop
for di = 1:length(f)
    k = 1;
    %% load data
    filename = fullfile('./data',f{di});
    outfilename = fullfile('./result',f{di});
    fn = [filename,'.mat'];
    data = load(fn);
    X = data.X;
    [~,n] = size(X);
    fea = Normalize_test(X,'norm');
    gnd = data.gnd;    
    clsnum = length(unique(gnd)); 
    Xn = fea;
    
    lambda_tlrr = lambda_tlrr_fileset(di);
    beta_tlrr = beta_tlrr_fileset(di);
    alpha_dplrr = alpha_dplrr_fileset(di);
    beta_dplrr = beta_dplrr_fileset(di);
    lambda_dplrr = 1/sqrt(log(n));
    lambda_lrr = lambda_lrr_fileset(di);
    lambda_sslrr = lambda_sslrr_fileset(di);
    lambda_lrpca = lambda_lrpca_fileset(di);
    lambda_cpssc = lambda_cpssc_fileset(di);
    lambda_sclrr = lambda_sclrr_fileset(di);
    beta_sclrr = beta_sclrr_fileset(di);
    lambda_clrr = lambda_clrr_fileset(di);
    
    for p = testset_p
        meanNum = 10;
        acc_rec1 = zeros(1,meanNum);
        acc_rec2 = zeros(1,meanNum);
        acc_rec3 = zeros(1,meanNum);
        acc_rec4 = zeros(1,meanNum);
        acc_rec5 = zeros(1,meanNum);
        acc_rec6 = zeros(1,meanNum);
        acc_rec7 = zeros(1,meanNum);
        acc_rec8 = zeros(1,meanNum);
        acc_rec9 = zeros(1,meanNum);
        nmi_rec1 = zeros(1,meanNum);
        nmi_rec2 = zeros(1,meanNum);
        nmi_rec3 = zeros(1,meanNum);
        nmi_rec4 = zeros(1,meanNum);
        nmi_rec5 = zeros(1,meanNum);
        nmi_rec6 = zeros(1,meanNum);
        nmi_rec7 = zeros(1,meanNum);
        nmi_rec8 = zeros(1,meanNum);
        nmi_rec9 = zeros(1,meanNum);
        parfor j = 1:meanNum
        disp(['idx=' num2str(j)])
        tmp = load([filename '_random_select_' num2str(p) '_' num2str(j) '.mat']);
        Omega_rand = tmp.Omega_rand;
        A = gnd2pair11(gnd, Omega_rand);    
        Omega = find(A~=0);
    %% low-rank representation (lrr) 
         [X,~,~,~,~] = lrr(Xn,Xn,lambda_lrr,opts);
         s = max(max(X));
         W_ans2 = 0.5*(abs(X) + abs(X'));
         W_ans2 = W_ans2 - diag(diag(W_ans2));
        idx_ans2 = SpectralClustering(W_ans2 ,clsnum);
        acc_rec2(j) = Accuracy(idx_ans2,double(gnd));
        [~, nmi_rec2(j), ~] = compute_nmi(double(gnd),idx_ans2);
    %% tensor low-rank representation
        [C,~,~,~,~] = tlrr_tnn_new(Xn,A,Omega,lambda_tlrr,beta_tlrr,s,opts); 
        Z_ans1 = C(:,:,1);
        B_ans1 = C(:,:,2);
        Z_ans1 = 0.5*(abs(Z_ans1) + abs(Z_ans1')); 
        Z_ans1 = Normalize_test(Z_ans1,'range'); 
        B_ans1 = B_ans1/s; 
        W_ans1 = FCSC(Z_ans1,B_ans1);  
        W_ans1 = 0.5*(W_ans1 + W_ans1');  
        W_ans1 = W_ans1 - diag(diag(W_ans1));  
        idx_ans1 = SpectralClustering(W_ans1 ,clsnum);
        acc_rec1(j) = Accuracy(idx_ans1,double(gnd));
        [~, nmi_rec1(j), ~] = compute_nmi(double(gnd),idx_ans1);
    %% dplrr
        [D0]= build_init_D(Xn,gnd,Omega_rand);
        [Z0_lrr]= build_lrr_Z0(gnd,Omega_rand,n);  
        [A_ans3 ,B_ans3, C_ans3, D_ans3, ~, Z_ans3]=...
            label_guided_lrr(Xn,D0,Z0_lrr,lambda_dplrr,alpha_dplrr,beta_dplrr);
        W_ans3 = (abs(Z_ans3)+abs(Z_ans3)')/2;
        W_ans3 = W_ans3 - diag(diag(W_ans3));
        idx_ans3 = SpectralClustering(W_ans3 ,clsnum);
        acc_rec3(j)= Accuracy(idx_ans3,double(gnd));
        [~, nmi_rec3(j), ~] = compute_nmi(double(gnd),idx_ans3);
    %% SSLRR
        [Z0_lrr]= build_lrr_Z0(gnd,Omega_rand,n);
        [~,~,Z_ans5] = sslrr(Xn,Z0_lrr,lambda_sslrr);
        Z_ans5(Z0_lrr==0) = 0;
        %Z_ans5(Z0_lrr==1) = 1;
         W_ans5 = 0.5*(abs(Z_ans5) + abs(Z_ans5'));
         W_ans5 = W_ans5 - diag(diag(W_ans5));
        idx_ans5 = SpectralClustering(W_ans5 ,clsnum);
        acc_rec5(j) = Accuracy(idx_ans5,double(gnd));
        [~, nmi_rec5(j), ~] = compute_nmi(double(gnd),idx_ans5);
    %% L-RPCA
        Y0 = init_label_matrix_lrpca(n,Omega_rand,gnd);
        [D,E_ans6,~] = l_rpca(Xn,Y0,lambda_lrpca,w,[]);
        W_ans6 = knngraph(D',n);
        idx_ans6 = SpectralClustering(W_ans6 ,clsnum);
        acc_rec6(j)= Accuracy(idx_ans6,double(gnd));
        [~, nmi_rec6(j), ~] = compute_nmi(double(gnd),idx_ans6);
            
      %% CP-SSC 
         [C_ans7,WF, err1J, err2J, err3J] = ...
             cp_ssc(Xn, A, false, lambda_cpssc);
         W_ans7 = C_ans7(1:n, :);
         W_ans7 = 0.5*(abs(W_ans7) + abs(W_ans7'));
         W_ans7 = W_ans7 - diag(diag(W_ans7));
        idx_ans7 = SpectralClustering(W_ans7 ,clsnum);
        acc_rec7(j)= Accuracy(idx_ans7, double(gnd));
         [~, nmi_rec7(j), ~] = compute_nmi(double(gnd),idx_ans7);
         
      %% SC-LRR
         [Z_ans8,E_ans8] = ...
             sclrr_v1(Xn, A, lambda_sclrr,beta_sclrr,[]);   
         W_ans8 = 0.5*(abs(Z_ans8) + abs(Z_ans8'));  
         W_ans8 = W_ans8 - diag(diag(W_ans8));
         idx_ans8 = SpectralClustering(W_ans8 ,clsnum);
         acc_rec8(j) = Accuracy(idx_ans8,double(gnd));
         [~, nmi_rec8(j), ~] = compute_nmi(double(gnd),idx_ans8);
         %% clrr        
         Q = init_mustlink_matrix(n,Omega_rand,gnd);
         [Z_ans9,~] = clrr(Xn,Q,lambda_clrr,opts);
         W_ans9 = Z_ans9 *Q';
         W_ans9 = 0.5*(abs( W_ans9) + abs( W_ans9'));
         W_ans9 = W_ans9 - diag(diag(W_ans9));
        idx_ans9 = SpectralClustering(W_ans9 ,clsnum);
        acc_rec9(j)= Accuracy(idx_ans9, double(gnd));
        [~, nmi_rec9(j), ~] = compute_nmi(double(gnd),idx_ans9);
        end
        acc_p_proposed(1,k) = mean(acc_rec1);
        acc_p_proposed(2,k) = std(acc_rec1);
        acc_p_lrr(1,k) = mean(acc_rec2);
        acc_p_lrr(2,k) = std(acc_rec2);
        acc_p_dplrr(1,k) = mean(acc_rec3);
        acc_p_dplrr(2,k) = std(acc_rec3);
        acc_p_sslrr(1,k) = mean(acc_rec5);
        acc_p_sslrr(2,k) = std(acc_rec5);
        acc_p_lrpca(1,k) = mean(acc_rec6);
        acc_p_lrpca(2,k) = std(acc_rec6);
        acc_p_cpssc(1,k) = mean(acc_rec7);
        acc_p_cpssc(2,k) = std(acc_rec7);
        acc_p_sclrr(1,k) = mean(acc_rec8);
        acc_p_sclrr(2,k) = std(acc_rec8);
        acc_p_clrr(1,k) = mean(acc_rec9);
        acc_p_clrr(2,k) = std(acc_rec9);
   
        nmi_p_proposed(1,k) = mean(nmi_rec1);
        nmi_p_proposed(2,k) = std(nmi_rec1);
        nmi_p_lrr(1,k) = mean(nmi_rec2);
        nmi_p_lrr(2,k) = std(nmi_rec2);
        nmi_p_dplrr(1,k) = mean(nmi_rec3);
        nmi_p_dplrr(2,k) = std(nmi_rec3);
        nmi_p_sslrr(1,k) = mean(nmi_rec5);
        nmi_p_sslrr(2,k) = std(nmi_rec5);
        nmi_p_lrpca(1,k) = mean(nmi_rec6);
        nmi_p_lrpca(2,k) = std(nmi_rec6);
        nmi_p_cpssc(1,k) = mean(nmi_rec7);
        nmi_p_cpssc(2,k) = std(nmi_rec7);
        nmi_p_sclrr(1,k) = mean(nmi_rec8);
        nmi_p_sclrr(2,k) = std(nmi_rec8);
        nmi_p_clrr(1,k) = mean(nmi_rec9);
        nmi_p_clrr(2,k) = std(nmi_rec9);
        k = k+1;
    end
     save([outfilename,'_demo.mat'],'testset_p',...
   'acc_p_proposed','acc_p_lrr','acc_p_dplrr','acc_p_sslrr',...
   'acc_p_lrpca','acc_p_cpssc','acc_p_sclrr','acc_p_clrr',...
   'nmi_p_proposed','nmi_p_lrr','nmi_p_dplrr','nmi_p_sslrr',...
   'nmi_p_lrpca','nmi_p_cpssc','nmi_p_sclrr','nmi_p_clrr');
end
delete(gcp('nocreate'));
toc
