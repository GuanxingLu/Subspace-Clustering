%% The comparision demo of our paper
% Semi-Supervised Subspace Clustering via Tensor Low-Rank Representation
% https://arxiv.org/abs/2205.10481
% before running the code, you need to download libraries from 
% https://github.com/GuanxingLu/Subspace-Clustering

clc,clear
addpath(genpath(cd))
% start parallel pool. If you run this script on your own desktop, just ignore this and modify "parfor" to "for"
delete(gcp('nocreate'));
parpool('local',12);

testset_p = 0.05:0.05:0.3;
acc_p_proposed = zeros(2,length(testset_p));
acc_p_lrr = acc_p_proposed;
nmi_p_proposed = zeros(2,length(testset_p));
nmi_p_lrr = acc_p_proposed;

f = {'ORL400_new';'YaleB944';'COIL20';'Isolet1';'2k2k_new';...
    'Alphabet';'BF0502';'Notting-Hill'};
%% Parameter settings
%Proposed
lambda_tlrr_fileset = [5 1 0.5 0.5 0.01 0.01 0.1 0.01];
beta_tlrr_fileset   = [0 0 10  10  10  10  10  10];
%LRR
lambda_lrr_fileset = [5 1 0.1 0.5 0.1 0.05 0.1 0.05];
opts.DEBUG = 1;
opts.tol = 1e-3;

%% file loop
tic
for di = [2 4 5 6 7]
    k = 1;
    %% load data
    filename = fullfile('./data',f{di});
    outfilename = fullfile('./result',f{di});
    fn = [filename,'.mat'];
    data = load(fn);
    X = data.X;
    [~,n] = size(X);
    fea = normalize(X,'norm');
    gnd = data.gnd;    
    clsnum = length(unique(gnd)); 
    Xn = fea;
    
    lambda_lrr = lambda_lrr_fileset(di);
    lambda_tlrr = lambda_tlrr_fileset(di);
    beta_tlrr = beta_tlrr_fileset(di);
    
    for p = testset_p
        meanNum = 10;
        acc_rec1 = zeros(1,meanNum);
        acc_rec2 = zeros(1,meanNum);
        nmi_rec1 = zeros(1,meanNum);
        nmi_rec2 = zeros(1,meanNum);
        parfor j = 1:meanNum
        disp(['idx=' num2str(j)])
        tmp = load([filename '_random_select_' num2str(p) '_' num2str(j) '.mat']);
        Omega_rand = tmp.Omega_rand;
        A = gnd2pair11(gnd, Omega_rand); % generate a -1,0,1 initial PCM
        Omega = find(A~=0); % initial set
    %% low-rank representation (lrr) 
        [X,~,~,~,~] = lrr(Xn,Xn,lambda_lrr,opts);
        W_ans2 = 0.5*(abs(X) + abs(X'));
        W_ans2 = W_ans2 - diag(diag(W_ans2));
        idx_ans2 = SpectralClustering(W_ans2 ,clsnum);
        acc_rec2(j) = Accuracy(idx_ans2,double(gnd));
        [~, nmi_rec2(j), ~] = compute_nmi(double(gnd),idx_ans2);
    %% tensor low-rank representation
        s = max(max(X));
        [C,~,~,~,~] = tlrr_tnn_new(Xn,A,Omega,lambda_tlrr,beta_tlrr,s,opts); 
        Z_ans1 = C(:,:,1);
        B_ans1 = C(:,:,2);
        Z_ans1 = 0.5*(abs(Z_ans1) + abs(Z_ans1')); 
        Z_ans1 = normalize(Z_ans1,'range'); 
        B_ans1 = B_ans1/s; 
        W_ans1 = FCSC(Z_ans1,B_ans1);  % repairment
        W_ans1 = 0.5*(W_ans1 + W_ans1');  
        W_ans1 = W_ans1 - diag(diag(W_ans1));  
        idx_ans1 = SpectralClustering(W_ans1 ,clsnum);
        acc_rec1(j) = Accuracy(idx_ans1,double(gnd));
        [~, nmi_rec1(j), ~] = compute_nmi(double(gnd),idx_ans1);
        end
        acc_p_proposed(1,k) = mean(acc_rec1);
        acc_p_proposed(2,k) = std(acc_rec1);
        acc_p_lrr(1,k) = mean(acc_rec2);
        acc_p_lrr(2,k) = std(acc_rec2);
   
        nmi_p_proposed(1,k) = mean(nmi_rec1);
        nmi_p_proposed(2,k) = std(nmi_rec1);
        nmi_p_lrr(1,k) = mean(nmi_rec2);
        nmi_p_lrr(2,k) = std(nmi_rec2);
        k = k+1;
    end
     save([outfilename,'_demo.mat'],'testset_p',...
   'acc_p_proposed','acc_p_lrr', 'nmi_p_proposed','nmi_p_lrr');
end
toc
delete(gcp('nocreate'));
