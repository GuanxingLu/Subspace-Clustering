%% This file also contains visual comparison of the affinity matrices learned by different methods
clc,clear
addpath(genpath(cd))
testset_p = 0.05:0.05:0.3;
err_p_rec = zeros(1,length(testset_p));
err_p_lrr = err_p_rec;
err_p_CSVT = err_p_rec;
err_p_kNNpairwise = err_p_rec;
err_p_sslrr = err_p_rec;
err_p_lrpca = err_p_rec;
err_p_cpssc = err_p_rec;
err_p_sclrr = err_p_rec;
err_p_clrr = err_p_rec;

nmi_p_rec = zeros(1,length(testset_p));
nmi_p_lrr = err_p_rec;
nmi_p_CSVT = err_p_rec;
nmi_p_kNNpairwise = err_p_rec;
nmi_p_sslrr = err_p_rec;
nmi_p_lrpca = err_p_rec;
nmi_p_cpssc = err_p_rec;
nmi_p_sclrr = err_p_rec;
nmi_p_clrr = err_p_rec;

f = {'usps400';'ORL400_new';'cotton';'YaleB944';'COIL20';...
    'dermatology_new';'Isolet1';'2k2k_new';'Alphabet'};
%% Parameter settings
%TLRR
lambda_tlrr_fileset = [0 5  0 1    0.01 0 0.5 0.01 0.01];
beta_tlrr_fileset   = [0 0.01 0 0.01 10   0 10 10 10];
%DPLRR
alpha_CSVT_fileset = [0 10 0 10   0.1  0 10 1 1 ];
beta_CSVT_fileset = [0  1  0 0.01 0.01 0 0.01 0.01 0.01];
%LRR
lambda_lrr_fileset = [0 5 0 1 0.1 0 0.5 0.1 0.05];
%SSLRR
%lambda_sslrr_fileset = [0.1 5 10 1 0.5 0.1 0.5 0.5];
%L-RPCA
w = 1;
lambda_lrpca_fileset = [0 0.1 0 1 0.05 0 0.5 1 0.05];   
%CPSSC
lambda_cpssc_fileset = [0 0.05 0 0.01 10 0 0.01 0.01 10] ;
%SCLRR
lambda_sclrr_fileset = [0 10 0 5 0.5 0 0.1 1 1];
beta_sclrr_fileset =   [0 1  0 1 1   0 0.1 1 1]; 
%CLRR
lambda_clrr_fileset = [0 5 0 1 0.5 0 0.5 0.1 0.1];

opts.DEBUG = 1;
opts.tol = 1e-3;
imageOpen = 0;
for di = [2 4 5 7 8 9]
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
    fea1 = fea;
    gnd1 = gnd;
    Xn = fea1; 
    
    lambda_tlrr = lambda_tlrr_fileset(di);
    beta_tlrr = beta_tlrr_fileset(di);
    
    alpha_CSVT = alpha_CSVT_fileset(di);
    beta_CSVT = beta_CSVT_fileset(di);
    
     lambda_lrr = lambda_lrr_fileset(di);
   %  lambda_sslrr = lambda_sslrr_fileset(di);
     
     %lambda_CSVT = lambda_lrr;
     lambda_CSVT = 1/sqrt(log(n));
     
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
       
        for j = 1:meanNum
        Omega_rand = load([filename '_random_select_' num2str(p) '_' num2str(j) '.mat']).Omega_rand;
        A = gnd2pair11(gnd1, Omega_rand);    
        Omega = find(A~=0);
    %% low rank representation (LRR)
         tic
            [X,~,~,~,~] = lrr(Xn,Xn,lambda_lrr,opts);
         toc
         s = max(max(X));
         W_ans2 = 0.5*(abs(X) + abs(X'));
         W_ans2 = W_ans2 - diag(diag(W_ans2));
         if  j==1  && p ==0.3 && imageOpen == 1
            figure
            imagesc(W_ans2);
             title('Matrix W(LRR))')
             colormap(jet);
            colorbar;
        end
        idx_ans2 = SpectralClustering(W_ans2 ,clsnum);
        acc_rec2(j)= Accuracy(idx_ans2,double(gnd1));
        [~, nmi_rec2(j), ~] = compute_nmi(double(gnd1),idx_ans2);
    %% tensor low rank representation (Proposed Method)
        tic
            [C,~,~,~,~] = tlrr_tnn(Xn,A,Omega,lambda_tlrr,beta_tlrr,s,opts);
        toc
         Z_ans1 = C(:,:,1);
         B_ans1 = C(:,:,2);
         B_ans1 = 0.5*(B_ans1+B_ans1');
         Z_ans1 = 0.5*(abs(Z_ans1) + abs(Z_ans1')); 
         Z_ans1 = Normalize_test(Z_ans1,'range');
         B_ans1 = B_ans1/s; 
         W_ans1 = FCSC(Z_ans1,B_ans1); 
         W_ans1 = 0.5*(W_ans1 + W_ans1');  
         W_ans1 = W_ans1 - diag(diag(W_ans1));  
         if  j==1  && p ==0.3 && imageOpen == 1
            figure
            imagesc(W_ans1);
             title('Matrix W(Proposed Method))')
             colormap(jet);
            colorbar;
         end
         idx_ans1 = SpectralClustering(W_ans1 ,clsnum);
         acc_rec1(j) = Accuracy(idx_ans1,double(gnd));
         [~, nmi_rec1(j), ~] = compute_nmi(double(gnd1),idx_ans1);
    %% DPLRR
            [D0]= build_init_D(Xn,gnd1,Omega_rand);  
            [Z0_lrr]= build_lrr_Z0(gnd1,Omega_rand,n);   
            tic
            [A_ans3 ,B_ans3, C_ans3, D_ans3, ~, Z_ans3]=...
            label_guided_lrr(Xn,D0,Z0_lrr,lambda_CSVT,alpha_CSVT,beta_CSVT);
            toc
            W_ans3 = (abs(Z_ans3)+abs(Z_ans3)')/2;
            W_ans3 = W_ans3 - diag(diag(W_ans3)); 
            if  j==1 && p ==0.3 && imageOpen == 1
                figure
                imagesc(W_ans3);
                 title('Matrix W(DPLRR))')
                 colormap(jet);
                colorbar;
            end
            idx_ans3 = SpectralClustering(W_ans3 ,clsnum);
            acc_rec3(j)= Accuracy(idx_ans3,double(gnd1));
            [~, nmi_rec3(j), ~] = compute_nmi(double(gnd1),idx_ans3);
    %% kNN
             tic
                Z_ans4 = kNNCompute(Xn,A);
             toc
             W_ans4 = 0.5*(abs(Z_ans4) + abs(Z_ans4'));
             W_ans4 = W_ans4 - diag(diag(W_ans4));
             if  j==1 && p ==0.3 &&  imageOpen == 1
                figure
                imagesc(W_ans4);
                 title('Matrix W(kNN))')
                 colormap(jet);
                colorbar;
            end
            idx_ans4 = SpectralClustering(W_ans4 ,clsnum);
            acc_rec4(j) = Accuracy(idx_ans4,double(gnd1));
            [~, nmi_rec4(j), ~] = compute_nmi(double(gnd1),idx_ans4);

    %% SSLRR
%             tic
%             [~,~,Z_ans5] = sslrr(Xn,Z0_lrr,lambda_sslrr);
%             toc
%             Z_ans5(Z0_lrr==0) = 0;
%             %Z_ans5(Z0_lrr==1) = 1;
%              W_ans5 = 0.5*(abs(Z_ans5) + abs(Z_ans5'));
%              W_ans5 = W_ans5 - diag(diag(W_ans5));
%              if  j==1 && p ==0.3 && imageOpen == 1
%                 figure
%                 imagesc(W_ans5);
%                  title('Matrix W(SSLRR))')
%                  colormap(jet);
%                 colorbar;
%             end
%             idx_ans5 = SpectralClustering(W_ans5 ,clsnum);
%             acc_rec5(j) = Accuracy(idx_ans5,double(gnd1));
    %% L-RPCA
            Y0 = init_label_matrix_lgx(n,Omega_rand,gnd1);
            tic
               % [D,E_ans6,~] = l_rpca(Xn,A,lambda_lrpca,w,[]);
                [D,E_ans6,~] = l_rpca(Xn,Y0,lambda_lrpca,w,[]);
            toc
            idx_ans6 = kmeans(D',clsnum,'Replicates',20);
            acc_rec6(j)= Accuracy(idx_ans6,double(gnd1));
            [~, nmi_rec6(j), ~] = compute_nmi(double(gnd1),idx_ans6);
            
   %% CP-SSC
     tic
        [C_ans7,WF, err1J, err2J, err3J] = cp_ssc(Xn, A, false, lambda_cpssc);
     toc
     W_ans7 = C_ans7(1:n, :);
     W_ans7 = 0.5*(abs(W_ans7) + abs(W_ans7'));
     W_ans7 = W_ans7 - diag(diag(W_ans7));
     if  j==1 && p ==0.3 &&  imageOpen == 1
            figure
            imagesc(W_ans7);
             title('Matrix W(CP-SSC)')
             colormap(jet);
            colorbar;
     end
     
        idx_ans7 = SpectralClustering(W_ans7 ,clsnum);
        acc_rec7(j)= Accuracy(idx_ans7, double(gnd1));
         [~, nmi_rec7(j), ~] = compute_nmi(double(gnd1),idx_ans7);
         
     %% SC-LRR
        tic
         [Z_ans8,E_ans8] = sclrr_v1(Xn, A, lambda_sclrr,beta_sclrr,[]);   
        toc
         W_ans8 = 0.5*(abs(Z_ans8) + abs(Z_ans8'));  
         W_ans8 = W_ans8 - diag(diag(W_ans8));  
         
         if  j==1 && p ==0.3 &&  imageOpen == 1
            figure
            imagesc(W_ans8);
             title('Matrix W(SC-LRR)')
             colormap(jet);
            colorbar;
         end
         
         idx_ans8 = SpectralClustering(W_ans8 ,clsnum);
         acc_rec8(j) = Accuracy(idx_ans8,double(gnd1));
          [~, nmi_rec8(j), ~] = compute_nmi(double(gnd1),idx_ans8);
         
          %% CLRR
           Q = init_label_matrix(n,Omega_rand,gnd1);
         tic
            [Z_ans9,~] = clrr(Xn,Q,lambda_clrr,opts);
         toc
         W_ans9 = Z_ans9 *Q';
         W_ans9 = 0.5*(abs( W_ans9) + abs( W_ans9'));
         W_ans9 = W_ans9 - diag(diag(W_ans9));

         if  j==1  && p ==0.3 &&  imageOpen == 1
                figure
                imagesc(W_ans9);
                 title('Matrix W(C-LRR)')
                 colormap(jet);
                colorbar;
         end
        idx_ans9 = SpectralClustering(W_ans9 ,clsnum);
        acc_rec9(j)= Accuracy(idx_ans9, double(gnd1));
       [~, nmi_rec9(j), ~] = compute_nmi(double(gnd1),idx_ans9);
         
        end
        err_p_rec(k) = mean(acc_rec1);
        err_p_lrr(k) = mean(acc_rec2);
        err_p_CSVT(k) = mean(acc_rec3);
        err_p_kNNpairwise(k) = mean(acc_rec4);
       % err_p_sslrr(k) = mean(acc_rec5);
        err_p_lrpca(k) = mean(acc_rec6);
        err_p_cpssc(k) = mean(acc_rec7);
        err_p_sclrr(k) = mean(acc_rec8);
        err_p_clrr(k) = mean(acc_rec9);
        
        nmi_p_rec(k) = mean(nmi_rec1);
        nmi_p_lrr(k) = mean(nmi_rec2);
        nmi_p_CSVT(k) = mean(nmi_rec3);
        nmi_p_kNNpairwise(k) = mean(nmi_rec4);
       % nmi_p_sslrr(k) = mean(nmi_rec5);
        nmi_p_lrpca(k) = mean(nmi_rec6);
        nmi_p_cpssc(k) = mean(nmi_rec7);
        nmi_p_sclrr(k) = mean(nmi_rec8);
        nmi_p_clrr(k) = mean(nmi_rec9);
        k = k+1;
    end
 save([outfilename,'_demo_LabelSelect_others.mat'],'testset_p',...
 'err_p_rec','err_p_lrr','err_p_CSVT','err_p_kNNpairwise',...
   'err_p_sslrr','err_p_lrpca','err_p_cpssc','err_p_sclrr','err_p_clrr',...
  'nmi_p_rec','nmi_p_lrr','nmi_p_CSVT','nmi_p_kNNpairwise',...
   'nmi_p_sslrr','nmi_p_lrpca','nmi_p_cpssc','nmi_p_sclrr','nmi_p_clrr');
end
