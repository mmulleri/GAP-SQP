% RI_Matejka16_figures.m
%-------------------------------------
% Generates figures for Section 4.1.
%-------------------------------------
% NOTE: Requires to run RI_Matejka16_main.m first.

clear;

fig_folder = 'Matejka16_figs/';
ours_text = 'GAP-SQP';
alt_text = 'Blahut-Arimoto';

%% Running times across information costs
% Figure 3.

load('Matejka16_output/data_runtimes_infocosts.mat')

set(groot,'DefaultAxesFontSize',16)

% Running times: SQP only
figure('PaperPosition',[0,0,8,5],'PaperSize',[8,5])
hold on

plot(llambda_grid_bench*10^3, times_mat(:,1), 'b-', 'LineWidth',2)

xlabel('Information cost \lambda \times 10^3')
ylabel('Seconds')
%title(['Running times for ' ours_text ' algorithm'])

figure_file = [fig_folder 'times_benchmark.pdf'];
print('-f1','-dpdf',figure_file)
close

% Running times: Compare
figure('PaperPosition',[0,0,8,5],'PaperSize',[8,5])
hold on

plot(llambda_grid_bench*10^3, times_mat(:,1), 'b-', 'LineWidth',2)
plot(llambda_grid_bench*10^3, times_mat(:,2), 'r-', 'LineWidth',2)
lgd = legend(ours_text,alt_text);
title(lgd,'Algorithm')

xlabel('Information cost \lambda \times 10^3')
ylabel('Seconds')
%title('Running times: Benchmark')

figure_file = [fig_folder 'times_benchmark_compare.pdf'];
print('-f1','-dpdf',figure_file)
close

%% Running times across grid precision points
% Figure 4

load('Matejka16_output/data_runtimes_gridpoints.mat')

% Running times: SQP
figure('PaperPosition',[0,0,8,5],'PaperSize',[8,5])
hold on

plot(gridpoints.^2*1e-3, times_mat(:,1), 'b-', 'LineWidth',2)

xlabel('Grid points (thousands)')
ylabel('Seconds')

figure_file = [fig_folder 'scale_SQP.pdf'];
print('-f1','-dpdf',figure_file)
close

% Running times: BA
figure('PaperPosition',[0,0,8,5],'PaperSize',[8,5])
hold on

plot(gridpoints.^2*1e-3, times_mat(:,2), 'r-', 'LineWidth',2)

xlabel('Grid points (thousands)')
ylabel('Seconds')

figure_file = [fig_folder 'scale_BA.pdf'];
print('-f1','-dpdf',figure_file)
close
