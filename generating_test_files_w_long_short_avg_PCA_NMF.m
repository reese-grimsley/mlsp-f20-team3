% This code is meant to be run as sections, not as an entire script

%% Import and get details from logfbank short window
clear
load('logfbank_renamed.mat');
logf= matfile('logfbank_renamed.mat');
dets = whos(logf);
dim = 26;

%% Import and get details from mfcc short window
clear
load('mfcc_renamed2.mat');
mfcc= matfile('mfcc_renamed.mat');
dets = whos(mfcc);
dim = 13;

%% Import and get details from logfbank long window
clear
load('logfbank_100ms.mat');
logf= matfile('logfbank_100ms.mat');
dets = whos(logf);
dim = 26;

%% Import and get details from mfcc long window
clear
load('mfcc_100ms.mat');
mfcc= matfile('mfcc_100ms.mat');
dets = whos(mfcc);
dim = 13;

%% Generate Readable Features
m = struct2table(dets);
names = cell2mat(table2array(m(:,1)));
 
[nSam, nChar] = size(names);
a_ncols = dim;

for i = 1:nSam
    a_test(:,:,i) = eval(names(i,1:10)); % store data into 3D array
    for j = 1:a_ncols
        a_sum(i,j) = mean(a_test(:,j,i)); % sum data across each frequency band
    end    
end

% Get Size
[nrows, ncols] = size(a_test(:,:,1));

%Make eigenface readable
for h = 1:nSam
      X(:,h) = reshape(a_test(:,:,h),nrows * ncols,1);
end

X_t = X'; % transpose so each row is a new sample

%% PCA - applied to full dataset (was unable to try)
[a_coeff, a_score, a_latent] = pca(X_t);

%% PCA - applied to reduced
[a_coeff, a_score, a_latent] = pca(a_sum);

%% NMF
[a_W, a_H] = nnmf(a_sum,a_ncols);

%% Get set mean to 0 and variance to 1 - PCA value
a_score_norm1 = normalize(a_score,'range',[-0.5 0.5]);

%% Get set mean to 0 and variance to 1 - NMF value
a_score_norm1 = normalize(a_W,'range',[-0.5 0.5]);

%% Get set mean to 0 and variance to 1 - normal value
a_score_norm1 = normalize(a_sum,'range',[-0.5 0.5]);


%% Ensure variance = 1 and mean = 0
a_max = max(a_score_norm1,[],'all');
a_min = min(a_score_norm1,[],'all');
a_mean = mean(a_score_norm1);
for n = 1:nSam
    for i = 1:50
        a_score_norm(n,i) = a_score_norm1(n,i) - a_mean(1,i); 
    end
end

a_mean1 = mean(a_score_norm); % test mean to make sure mean = 0 

% run the following sections depending on which data you constructed

%% write logfbank short window data
writematrix(a_score_norm,'logfbank_short_test.csv');

%% write mfcc short window data
writematrix(a_score_norm,'mfcc_short_test.csv');

%% write logfbank short window NMF data
writematrix(a_score_norm,'logfbank_short_NMF_test.csv');

%% write mfcc short window NMF data
writematrix(a_score_norm,'mfcc_short_NMF_test.csv');

%% write logfbank short window PCA data
writematrix(a_score_norm,'logfbank_short_PCA_test.csv');

%% write mfcc short window PCA data
writematrix(a_score_norm,'mfcc_short_PCA_test.csv');

%% write logfbank long window data
writematrix(a_score_norm,'logfbank_long_test.csv');

%% write mfcc long window data
writematrix(a_score_norm,'mfcc_long_test.csv');

%% write logfbank long window NMF data
writematrix(a_score_norm,'logfbank_long_NMF_test.csv');

%% write mfcc long window NMF data
writematrix(a_score_norm,'mfcc_long_NMF_test.csv');

%% write logfbank long window PCA data
writematrix(a_score_norm,'logfbank_long_PCA_test.csv');

%% write mfcc long window PCA data
writematrix(a_score_norm,'mfcc_long_PCA_test.csv');

%%

% Once these .csv's are exported, the data is futher manipulated using
% Excel. This is done by labeling each row to the appropriate sample, 
% deleting data points that are not included in the testing procedure. Then
% applying those data points to the STC feature .csv's Reese generate. This
% created the readable .csv's that were using for testing.
