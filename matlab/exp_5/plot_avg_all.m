
clear
close all

eval_plot = 'fmr_0d01'; % eer fmr_1 fmr_0d1 fmr_0d01 fmr_0
data_class = 'ethnicities'; % overall genders ethnicities classes

% Load data
data = open([fullfile(fullfile(cd, '..'), '..') ...
        '/FaceRecognitionPython_data_store/Result/average_exp_result_mat/exp_5/exp_5_avg_' data_class '.mat']);

data_classes = strtrim(string(data.data_classes));
if strcmp(data_class, 'overall')

elseif strcmp(data_class, 'genders')
    model_series = [data.avg_scores.fmr_0d01.female; data.avg_scores.fmr_0d01.male];
    model_error = [data.std_scores.fmr_0d01.female; data.std_scores.fmr_0d01.male];
    model_sumranked = [data.sum_ranked.fmr_0d01.female; data.sum_ranked.fmr_0d01.male];
elseif strcmp(data_class, 'ethnicities')
    model_series = [data.avg_scores.fmr_0d01.female; data.avg_scores.fmr_0d01.male];
    model_error = [data.std_scores.fmr_0d01.female; data.std_scores.fmr_0d01.male];
    model_sumranked = [data.sum_ranked.fmr_0d01.female; data.sum_ranked.fmr_0d01.male];
elseif strcmp(data_class, 'classes')

end
    
% Assign algorithm names
algo_names = strtrim(string(data.algo_names));
for i = 1 : numel(algo_names)
    if strcmp(algo_names(i), 'euclid')
        algo_names(i) = 'Euclidean';
    elseif strcmp(algo_names(i), 'selmEuclidDistPOS')
        algo_names(i) = 'SELM(distance)';
    elseif strcmp(algo_names(i), 'selmEuclidMeanPOS')
        algo_names(i) = 'SELM(mean)';
    elseif strcmp(algo_names(i), 'selmEuclidMultiplyPOS')
        algo_names(i) = 'SELM(multiply)';
    elseif strcmp(algo_names(i), 'selmEuclidSumPOS')
        algo_names(i) = 'SELM(sum)';
    end
end

legend_loc = 'northeast';
if strcmp(eval_plot, 'eer')
    eval_label = 'EER';
elseif strcmp(eval_plot, 'fmr_1')
    eval_label = 'FMR 1%';
elseif strcmp(eval_plot, 'fmr_0d1')
    eval_label = 'FMR 0.1%';
elseif strcmp(eval_plot, 'fmr_0d01')
    eval_label = 'FMR 0.01%';
elseif strcmp(eval_plot, 'fmr_0')
    eval_label = 'FMR 0%';
end

% Plot
figure('Position', [1 1 882 704]);
b = bar(model_series, 'grouped');
%%For MATLAB R2019a or earlier releases
hold on
% Find the number of groups and the number of bars in each group
ngroups = size(model_series, 1);
nbars = size(model_series, 2);
% Calculate the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, model_series(:,i), model_error(:,i), 'k', 'linestyle', 'none');
end
hold off
%%For MATLAB 2019b or later releases
hold on
% Calculate the number of bars in each group
nbars = size(model_series, 2);
% Get the x coordinate of the bars
x = [];
for i = 1:nbars
    x = [x ; b(i).XEndPoints];
end
% Plot the errorbars
errorbar(x',model_series,model_error,'k','linestyle','none');
hold off

xticklabels(strtrim(string(data.data_classes)));
set(gca,'fontname','times new roman')
set(gca,'FontSize', 14)
xlabel('Classes', 'fontname', 'times new roman', 'fontSize', 18)
ylabel(eval_label, 'fontname', 'times new roman', 'fontSize', 18)

legend(algo_names, 'Location', legend_loc, 'fontname', 'times new roman', 'fontSize', 16)

% Text over bar
hT = [];
for i = 1 : length(b)
  hT = [hT, text(b(i).XData+b(i).XOffset, ...
      b(i).YData, ...
      num2str(model_sumranked(:,i)), ...
      'VerticalAlignment', 'bottom', 'horizontalalign', 'center', ...
      'fontname', 'times new roman', 'fontSize', 16)];
end





