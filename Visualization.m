%% Comparisons of the clustering accuracy and NMI of different methods.
clc,clear
close all
addpath(genpath(cd))
f = {'usps400';'ORL400_new';'cotton';...
    'YaleB944';'COIL20';'dermatology_new';'Isolet1';'2k2k_new';'Alphabet'};
f1 = {'usps400';'ORL';'cotton';'YaleB';'COIL20';...
    'dermatology_new';'Isolet';'MNIST';'Alphabet'};
for di = [4 5 7 8 9]
dataname = f{di};
filename = fullfile('./result',dataname);
fn = [filename,'_demo_LabelSelect_others.mat'];
Data = load(fn);
fn2 = [filename,'_sslrr_1130_LabelSelect.mat'];
Data2 = load(fn2);
%fn3 = [filename,'_demo_LabelSelect_dplrr.mat'];
%Data3 = load(fn3);
fn4 = [filename,'_demo_lrpca.mat'];
Data4 = load(fn4);
fn_tnn = [filename,'_demo_tnn_0120.mat'];
Data_tnn = load(fn_tnn);

figure
p1 = plot(100*Data_tnn.testset_p,Data_tnn.err_p_rec,'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.err_p_lrr,'--^','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.err_p_CSVT,':d','LineWidth',1.2);
hold on
%plot(100*Data3.testset_p,Data3.err_p_CSVT,':d','LineWidth',1.2);
%hold on
plot(100*Data2.testset_p,Data2.acc_p_sslrr,'-.*','LineWidth',1.2);
hold on
plot(100*Data4.testset_p,Data4.err_p_lrpca,'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.err_p_cpssc,'--<','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.err_p_sclrr,':s','LineWidth',1.2);   
hold on
plot(100*Data.testset_p,Data.err_p_clrr,'-.o','LineWidth',1.2);
p1(1).Color = [0.894117647058824 0.121568627450980 0.149019607843137];
xlabel('Labeled Percentage (%)','FontSize',26)
ylabel('Accuracy (%)','FontSize',26)
%legend('Proposed Method','LRR','DPLRR','SSLRR','L-RPCA','CP-SSC',...
%'SC-LRR','CLRR','NumColumns',2,'FontSize',11)
%legend boxon;
title([f1{di}],'FontSize',26)

figure
p11 = plot(100*Data_tnn.testset_p,Data_tnn.nmi_p_rec,'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_lrr,'--^','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_CSVT,':d','LineWidth',1.2);
hold on
%plot(100*Data3.testset_p,Data3.nmi_p_CSVT,':d','LineWidth',1.2);
%hold on
plot(100*Data2.testset_p,Data2.nmi_p_sslrr,'-.*','LineWidth',1.2);
hold on
plot(100*Data4.testset_p,Data4.nmi_p_lrpca,'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_cpssc,'--<','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_sclrr,':s','LineWidth',1.2);   
hold on
plot(100*Data.testset_p,Data.nmi_p_clrr,'-.o','LineWidth',1.2);
p11(1).Color = [0.894117647058824 0.121568627450980 0.149019607843137];
xlabel('Labeled Percentage (%)','FontSize',26)
ylabel('NMI','FontSize',26)
%set(gca,);
%lgd.NumColumns = 2;
title([f1{di}],'FontSize',26)
end

%% Iteration curve
%clc,clear
%close all
%addpath(genpath(cd))
f = {'usps400';'ORL400_new';'cotton';...
    'YaleB944';'COIL20';'dermatology_new';'Isolet1';'2k2k_new';'Alphabet'};
figure
for di = [2 4 5 7 8 9]
    dataname = f{di};
    filename = fullfile('./result',dataname);
    fn = [filename,'_tnn_ERR_0124_1.mat'];
    Data = load(fn);
    plot(1:Data.maxiter,Data.ERR/max(Data.ERR),'-','LineWidth',1.8);
    hold on
end
xlabel('Number of Iteration','FontSize',14)
ylabel('Convergence Criterion','FontSize',14)
legend('ORL','YaleB','COIL20','Isolet','MNIST','Alphabet','FontSize',11)
