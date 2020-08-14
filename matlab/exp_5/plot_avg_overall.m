clear
close all

eval_plot = 'fmr_1'; % eer fmr_1 fmr_0d1 fmr_0d01 fmr_0
data_class = 'overall';

% Load data
data = open([fullfile(fullfile(cd, '..'), '..') ...
    '/FaceRecognitionPython_data_store/Result/average_exp_result_mat/exp_5/exp_5_avg_overall.mat']);

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
legend_loc = 'southeast';
if strcmp(eval_plot, 'eer')
    eval_label = 'EER';
    my_ylim = [1.25, 1.55];
elseif strcmp(eval_plot, 'fmr_1')
    legend_loc = 'northeast';
    eval_label = 'FMR 1%';
    my_ylim = [1.45, 1.85];
elseif strcmp(eval_plot, 'fmr_0d1')
    eval_label = 'FMR 0.1%';
    my_ylim = [2.5, 3.2];
elseif strcmp(eval_plot, 'fmr_0d01')
    eval_label = 'FMR 0.01%';
    my_ylim = [3.4, 5.8];
elseif strcmp(eval_plot, 'fmr_0')
    eval_label = 'FMR 0%';
    my_ylim = [3.5, 8.5];
end

% Plot
figure('Position', [1 1 882 704])
hAx = gca;
hB = bar(data.scores.(eval_plot).(data_class));
set(gca,'fontname','times new roman')
set(gca,'FontSize', 14)
% hAx.YAxis.FontSize = 14;
% hAx.XAxis.FontSize = 14;
xlabel('Experiment', 'fontname', 'times new roman', 'fontSize', 18)
ylabel(eval_label, 'fontname', 'times new roman', 'fontSize', 18)
grid on
grid minor
legend(algo_names, 'Location', legend_loc, 'fontname', 'times new roman', 'fontSize', 16)
ylim(my_ylim)

% Text over bar
hT = [];
for i = 1 : length(hB)
  hT = [hT, text(hB(i).XData+hB(i).XOffset, ...
      hB(i).YData, ...
      num2str(data.ranked.(eval_plot).(data_class)(:,i)), ...
      'VerticalAlignment', 'bottom', 'horizontalalign', 'center', ...
      'fontname', 'times new roman', 'fontSize', 16)];
end


