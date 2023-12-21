%% Visualize some samples of the datasets we used
clc,clear
close all
addpath(genpath(cd))

f = {'ORL400_new';'YaleB944';'COIL20';'Isolet1';'2k2k_new';...
    'Alphabet';'BF0502';'Notting-Hill'};
f1 = {'ORL';'YaleB';'COIL20';'Isolet';'MNIST';...
    'Alphabet';'BF0502';'Notting-Hill'};
rng(42)
%% file loop
tic
for di = 1:8
    k = 1;
    %% load data
    filename = fullfile('./data',f{di});
    outfilename = fullfile('./result',f{di});
    fn = [filename,'.mat'];
    data = load(fn);
    X = data.X;
    [d,n] = size(X);
    gnd = data.gnd;    
    clsnum = length(unique(gnd)); 
    
    figure;
    rand_indices = randperm(n, 64); % randomly select 64 samples
    for i = 1:16
        subplot(4, 4, i);
        img = reshape(X(:, rand_indices(i)), sqrt(d), sqrt(d)); % reshape sample to image format
        imshow(img, []);
        title(sprintf('Class %d', gnd(rand_indices(i))), 'FontSize', 10);
    end
%     sgtitle(sprintf('Random samples from %s dataset', f1{di}));
    set(gcf, 'PaperPositionMode', 'auto');
%     saveas(gcf, fullfile(outfilename, sprintf('%s_samples.png', f{di})));
   
end
toc