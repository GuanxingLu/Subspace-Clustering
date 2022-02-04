%% before running the code, you need download the LinADMM library from:https://github.com/canyilu/LibADMM
%iteration
clc,clear
addpath(genpath(cd))
testset_p = 0.3;
err_p_rec = zeros(1,length(testset_p));
nmi_p_rec = err_p_rec;
err_p_lrr = err_p_rec;
nmi_p_lrr = err_p_rec;
err_p_cpssc = err_p_rec;
nmi_p_cpssc = err_p_rec;
err_p_sclrr = err_p_rec;
nmi_p_sclrr = err_p_rec;
err_p_clrr = err_p_rec;
nmi_p_clrr = err_p_rec;
f = {'usps400';'ORL400_new';'cotton';'YaleB944';'COIL20';...
    'dermatology_new';'Isolet1';'2k2k_new';'Alphabet'};
f1 = {'usps400';'ORL';'cotton';'YaleB';'COIL20';...
    'dermatology_new';'Isolet';'MNIST';'Alphabet'};
%% Parameter settings
%Proposed Method
lambda_tlrr_fileset = [0 5  0 1    0.01 0 0.5 0.01 0.01];
beta_tlrr_fileset   = [0 0.01 0 0.01 10   0 10 10 10];
%LRR
lambda_lrr_fileset = [0 5 0 1 0.1 0 0.5 0.1 0.05];
opts.DEBUG = 1;
opts.tol = 1e-3;
imageOpen = 1;
figure
for di = [2 4 5 7 8 9]
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
    fea1 = fea;
    gnd1 = gnd;
    Xn = fea1;  
    
    lambda_tlrr = lambda_tlrr_fileset(di);
    beta_tlrr = beta_tlrr_fileset(di);

     lambda_lrr = lambda_lrr_fileset(di);
    for p = testset_p
        meanNum = 1;
        acc_rec1 = zeros(1,meanNum);
        nmi_rec1 = acc_rec1;
        acc_rec2 = acc_rec1;
        nmi_rec2 = acc_rec1;
        for j = 1:meanNum
        Omega_rand = load([filename '_random_select_' num2str(p) '_' num2str(j) '.mat']).Omega_rand;
        A = gnd2pair11(gnd1, Omega_rand);    
        Omega = find(A~=0);

    %% low rank representation (lrr) 
         tic
            [X,~,~,ERR,maxiter] = lrr(Xn,Xn,lambda_lrr,opts);
         toc
         s = max(max(X));
         W_ans2 = 0.5*(abs(X) + abs(X'));
         W_ans2 = W_ans2 - diag(diag(W_ans2));
        idx_ans2 = SpectralClustering(W_ans2 ,clsnum);
        acc_rec2(j)= Accuracy(idx_ans2,double(gnd1));
       [~, nmi_rec2(j), ~] = compute_nmi(double(gnd1),idx_ans2);
    %% tensor low rank representation
         tic
         [C,~,~,ERR,maxiter] = tlrr_tnn_new(Xn,A,Omega,lambda_tlrr,beta_tlrr,s,opts); 
         toc
         Z_ans1 = C(:,:,1);
          B_ans1 = C(:,:,2);
          B_ans1 = 0.5*(B_ans1+B_ans1');
          Z_ans1 = 0.5*(abs(Z_ans1) + abs(Z_ans1')); 
          Z_ans1 = normalize(Z_ans1,'range'); 
          B_ans1 = B_ans1/s; 
          W_ans1 = FCSC(Z_ans1,B_ans1);  
         W_ans1 = 0.5*(W_ans1 + W_ans1');  
         W_ans1 = W_ans1 - diag(diag(W_ans1));  
         idx_ans1 = SpectralClustering(W_ans1 ,clsnum);
         acc_rec1(j) = Accuracy(idx_ans1,double(gnd1));
         [~, nmi_rec1(j), ~] = compute_nmi(double(gnd1),idx_ans1);
        end
        err_p_rec(k) = mean(acc_rec1);
        nmi_p_rec(k) = mean(nmi_rec1);
        err_p_lrr(k) = mean(acc_rec2);
        nmi_p_lrr(k) = mean(nmi_rec2);
        k = k+1;
    end
  %  save([outfilename,'_demo_tnn_0120.mat'],'testset_p','err_p_rec',...
  %  'err_p_lrr','nmi_p_rec','nmi_p_lrr');
     save([outfilename,'_tnn_ERR_0124_2.mat'],'testset_p','maxiter','ERR',...
     'err_p_rec','err_p_lrr','nmi_p_rec','nmi_p_lrr');
%% Iteration curve
    plot(1:maxiter,ERR,'-','LineWidth',1.8);
    hold on
    %xlabel('Number of Iteration','FontSize',26)
    %ylabel('Convergence Criterion','FontSize',26)
    %title([f1{di}],'FontSize',26)
end
xlabel('Number of Iteration','FontSize',26)
ylabel('Convergence Criterion','FontSize',26)
legend('ORL','YaleB','COIL20','Isolet','Mnist','Alphabet','FontSize',15)
%title([f1{di}],'FontSize',26)
