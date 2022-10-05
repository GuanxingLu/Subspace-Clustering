%% Comparisons of the clustering accuracy and NMI of different methods.
clc,clear
close all
addpath(genpath(cd))
f = {'ORL400_new';'YaleB944';'COIL20';'Isolet1';'2k2k_new';...
    'Alphabet';'BF0502';'Notting-Hill'};
f1 = {'ORL';'YaleB';'COIL20';'Isolet';'MNIST';...
    'Alphabet';'BF0502';'Notting-Hill'};
for di = 1:8
dataname = f{di};
filename = fullfile('./result',dataname);
fn = [filename,'_demo.mat'];
Data = load(fn);

figure
p1 = plot(100*Data.testset_p,Data.acc_p_proposed(1,:),'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.acc_p_lrr(1,:),'--^','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.acc_p_dplrr(1,:),':d','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.acc_p_sslrr(1,:),'-.*','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.acc_p_lrpca(1,:),'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.acc_p_cpssc(1,:),'--<','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.acc_p_sclrr(1,:),':s','LineWidth',1.2);   
hold on
plot(100*Data.testset_p,Data.acc_p_clrr(1,:),'-.o','LineWidth',1.2);
p1(1).Color = [0.894117647058824 0.121568627450980 0.149019607843137];
xlabel('Labeled Percentage (%)','FontSize',26)
ylabel('Accuracy','FontSize',26)
if di == 1
    legend('Proposed Method','LRR','DPLRR','SSLRR','L-RPCA','CP-SSC',...
    'SC-LRR','CLRR','NumColumns',2,'FontSize',11)
end
title([f1{di}],'FontSize',26)

% NMI
figure
p11 = plot(100*Data.testset_p,Data.nmi_p_proposed(1,:),'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_lrr(1,:),'--^','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_dplrr(1,:),':d','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_sslrr(1,:),'-.*','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_lrpca(1,:),'->','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_cpssc(1,:),'--<','LineWidth',1.2);
hold on
plot(100*Data.testset_p,Data.nmi_p_sclrr(1,:),':s','LineWidth',1.2);   
hold on
plot(100*Data.testset_p,Data.nmi_p_clrr(1,:),'-.o','LineWidth',1.2);
p11(1).Color = [0.894117647058824 0.121568627450980 0.149019607843137];
xlabel('Labeled Percentage (%)','FontSize',26)
ylabel('NMI','FontSize',26)
title([f1{di}],'FontSize',26)
end
