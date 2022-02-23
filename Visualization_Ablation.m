clc,clear
close all
addpath(genpath(cd))
testset_p = 0.05:0.05:0.3;
err_p_rec = zeros(1,length(testset_p));
nmi_p_rec = err_p_rec;
err_p_lrr = err_p_rec;
nmi_p_lrr = err_p_rec;
err_p_woR = err_p_rec;
nmi_p_woR = err_p_rec;
err_p_woL =  err_p_rec;
nmi_p_woL = err_p_rec;
f = {'usps400';'ORL400_new';'cotton';'YaleB944';'COIL20';...
    'dermatology_new';'Isolet1';'2k2k_new';'Alphabet'};
f1 = {'usps400';'ORL';'cotton';'YaleB';'COIL20';...
    'dermatology_new';'Isolet';'MNIST';'Alphabet'};
for di = 7
    k = 1;
    %% load data
    filename = fullfile('./data',f{di});
    outfilename = fullfile('./result',f{di});
    ablationStudy = load([outfilename,'_ablationStudy_0221.mat'],'testset_p','err_p_rec',...
   'nmi_p_rec','err_p_woR','nmi_p_woR','err_p_woL','nmi_p_woL');
    ablationStudyACC = load([outfilename,'_ablationStudy_0221.mat'],'testset_p','err_p_rec',...
   'err_p_woR','err_p_woL');
    ablationStudyNMI = load([outfilename,'_ablationStudy_0221.mat'],'testset_p',...
   'nmi_p_rec','nmi_p_woR','nmi_p_woL');

    figure
    plot(100*testset_p,ablationStudy.err_p_rec,'->','LineWidth',1.2);
    hold on
    plot(100*testset_p,ablationStudy.err_p_woR,'-s','LineWidth',1.2);
    hold on
    plot(100*testset_p,ablationStudy.err_p_woL,'-s','LineWidth',1.2);
    hold on
    legend('Origin','woR','woL');
    xlabel('Labeled Percentage (%)')
    ylabel('Accuracy (%)');
    title([f1{di}])
    
    figure
    plot(100*testset_p,ablationStudy.nmi_p_rec,'->','LineWidth',1.2);
    hold on
    plot(100*testset_p,ablationStudy.nmi_p_woR,'-s','LineWidth',1.2);
    hold on
    plot(100*testset_p,ablationStudy.nmi_p_woL,'-s','LineWidth',1.2);
    hold on
    legend('Origin','woR','woL');
    xlabel('Labeled Percentage (%)')
    ylabel('NMI');
    title([f1{di}])
end
%%
clc,clear
mean([.9337  .9548  .9716  .9218 .7825  .8107])