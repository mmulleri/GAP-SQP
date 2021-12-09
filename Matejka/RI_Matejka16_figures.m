% RI_Matejka16_figures.m
%-------------------------------------
% Generates figures fr Section 4.1.
%-------------------------------------
% NOTE: Requires to run RI_Matejka16_main.m first.

clear;

fig_folder = 'Matejka16_figs/';
ours_text = 'GAP-SQP';
alt_text = 'Blahut-Arimoto';

%% Running times across information costs
% Figure 3.

load('Matejka16_output/data_runtimes_infocosts.mat')

dblue='#143D73';
lblue='#96AFC2';
dorange='#F29F05';
dred='#BF214B';

% Running times: SQP only
fig=figure(1);
clf
fig.Units = 'inches';
fig.Position = [0 0 3 2.1];
hold on

plot(llambda_grid_bench*10^3, times_mat(:,1), '-', 'LineWidth',2,'Color',dblue)

xlabel('information cost \lambda \times 10^3')
ylabel('seconds')
%title(['Running times for ' ours_text ' algorithm'])

set(gca,'FontSize',9);
set(gca,'FontName','CMU Serif');
figure_file = [fig_folder 'times_benchmark.pdf'];
exportgraphics(fig,figure_file,'ContentType','vector');

% Running times: Compare
cla
hold on

plot(llambda_grid_bench*10^3, times_mat(:,1), '-', 'LineWidth',2,'Color',dblue)
plot(llambda_grid_bench*10^3, times_mat(:,2), '-', 'LineWidth',2,'Color',dorange)
lgd = legend(ours_text,alt_text);
%title(lgd,'Algorithm')
lgd.Location = 'north west';

figure_file = [fig_folder 'times_benchmark_compare.pdf'];
exportgraphics(fig,figure_file,'ContentType','vector');

%% Objective function comparison
% Figure 10
cla
fig.Position = [0 0 6 3];
hold on
plot(llambda_grid_bench*10^3, obj_mat(:,1)-obj_mat(:,2),'-','LineWidth',1,'Color',dblue)

xlabel('information cost \lambda \times 10^3')
ylabel('difference in objective value')
lgd.Visible = 'off';

figure_file = [fig_folder 'obj_benchmark.pdf'];
exportgraphics(fig,figure_file,'ContentType','vector');
fig.Position = [0 0 3 2];

%% Running times across grid precision points
% Figure 4

load('Matejka16_output/data_runtimes_gridpoints.mat')

% Running times: GAP_SQP
cla
hold on

plot(gridpoints.^2*1e-3, times_mat(:,1), '-', 'LineWidth',2,'Color',dblue)

xlabel('grid points (thousands)')
ylabel('seconds')

figure_file = [fig_folder 'scale_SQP.pdf'];
exportgraphics(fig,figure_file,'ContentType','vector');

% Running times: BA
cla
hold on

plot(gridpoints.^2*1e-3, times_mat(:,2), '-', 'LineWidth',2,'Color',dorange)

figure_file = [fig_folder 'scale_BA.pdf'];
exportgraphics(fig,figure_file,'ContentType','vector');
close

