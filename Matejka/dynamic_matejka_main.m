% dynamic_matejka_main.m
%-------------------------------------
% This script solves the dynamic RI tracking problem for the QE revision.
%-------------------------------------

%Initialize folders and path.
clear;
clc;

%% Add path

addpath('..');

%% Folders

% Numerical output
if ~exist('Matejka16_output', 'dir')
    mkdir('Matejka16_output')
end

% Figures
if ~exist('Matejka16_figs', 'dir')
    mkdir('Matejka16_figs')
end

% Compares running times between the MX approach and
% a variant that substitutes GAP-SQP for the BA step.

%% Model parameters

% Constant settings - Model
ufun = @(p,d) p^(-(d+1)/d)*(p-1);
staterange = [1/9 1/2];
actionrange = [10/9 3/2];

% Information cost
llambda = 0.00531;
% Time discount factor
bbeta = 1.0;


%% Computation parameters

% Destination folder
d_folder = 'Matejka16_output/';
stats = zeros(0,12);
comparison = zeros(0,3);

% Number of time periods
T = 10;

% Runs:
Rho = .7;%.8:-.1:.6;%1:-.1:0;
N = size(Rho,2);
NA = 100*ones(1,N);
NS = 300*ones(1,N);

%% Run times across settings
for i=1:N

    LA = NA(i); % Number of actions
    LS = NS(i); % Number of states
    rrho = Rho(i); % AR(1) persistence

    fprintf('Comparison %i (%i actions, %i states, %i periods, rho = %f):\n',i,LA,LS,T,rrho);

    %% Setup
    % Grid over demand elasticities (state) and prices (action)
    stategrid = transpose(staterange(1):(staterange(2)-staterange(1))/(LS-1):staterange(2));
    actiongrid = transpose(actionrange(1):(actionrange(2)-actionrange(1))/(LA-1):actionrange(2));


    % Flow payoffs u(a,s)
    u_mat = zeros(LA,LS);

    for ia=1:LA
        for is=1:LS
            u_mat(ia,is) = ufun(actiongrid(ia),stategrid(is));
        end
    end


    % % Exogenous Markov state transition
    % % ppi_trans(s|s_old)
    % ppi_trans = rrho * eye(LS)+(1-rrho) * 1/LS * ones(LS,LS);
    % % Initial prior mmu_init:
    % mmu_init = 1/LS * ones(LS,1);

    % tauchenized version:
    mean_state =(staterange(1)+staterange(2))/2.0;
    stds = 3.0; % # of Standard deviations over which to Tauchenize
    uncond_vol = (staterange(2)-staterange(1))/(2*stds); % Unconditional volatility of state
    innov_vol = uncond_vol*sqrt(1-rrho^2); % Volatility of AR(1) innovation
    ppi_trans = zeros(LS,LS);
    wwindow = stategrid(2)-stategrid(1);
    stategrid_bounds_temp = stategrid-wwindow/2;
    stategrid_bounds = [stategrid_bounds_temp; stategrid_bounds_temp(end)+wwindow];
    for s_old=1:LS
        for s=1:LS
            ppi_trans(s,s_old) = normcdf(stategrid_bounds(s+1),rrho*stategrid(s_old)+(1-rrho)*mean_state,innov_vol) - ...
                normcdf(stategrid_bounds(s),rrho*stategrid(s_old)+(1-rrho)*mean_state,innov_vol);
        end
        ppi_trans(:,s_old) = ppi_trans(:,s_old)/sum(ppi_trans(:,s_old));
    end

    mmu_init = zeros(LS,1);
    for is=1:LS
        mmu_init(is) = normcdf(stategrid_bounds(is+1),mean_state,uncond_vol) - ...
            normcdf(stategrid_bounds(is),mean_state,uncond_vol);
    end
    mmu_init = mmu_init/sum(mmu_init);

    [stats_new, comp_new] = dynamic_timed_runs(u_mat,mmu_init,ppi_trans,llambda,bbeta,T);
    stats = [ stats ; i*ones(size(stats_new,1),1), stats_new ];
    comparison = [ comparison ; i, comp_new ];
end


save('Matejka16_output/runtimes_rho')

%% Figures

dblue='#143D73';
lblue='#96AFC2';
dorange='#F29F05';
dred='#BF214B';

fig_folder = 'Matejka16_figs/';
ours_text = 'GAP-SQP';
alt_text = 'Blahut-Arimoto';

% Comparison

cla
fig=figure(1);
for i=1:N
    GAP_stats_i= stats(stats(:,1)==i & stats(:,2)==1,3:end);
    BA_stats_i = stats(stats(:,1)==i & stats(:,2)==0,3:end);

    fprintf('Rho = %3g. Average duration of an iteration: %6gs with GAP, %6gs with BA subroutine.\n',Rho(i),GAP_stats_i(end,1)/GAP_stats_i(end,2),BA_stats_i(end,1)/BA_stats_i(end,2))

    clf
    fig.Units = 'inches';
    fig.Position = [0 0 3 2.1];

    xlabel('running time in s')
    ylabel('welfare')
    %title(sprintf('Welfare comparison ($\\rho=%3g$)',Rho(i)),'Interpreter','latex');
    set(gca,'FontSize',9);
    set(gca,'FontName','CMU Serif');
    hold on
    plot(GAP_stats_i(:,1), GAP_stats_i(:,3), '-', 'LineWidth',2,'Color',dblue)
    plot(BA_stats_i(:,1), BA_stats_i(:,3), '-', 'LineWidth',2,'Color',dorange)
    lgd = legend(ours_text,alt_text);
    %title(lgd,'Algorithm')
    lgd.Location = 'south east';

    figure_file = sprintf('%sdynamic_welfare_comparison_%1.1f.pdf',fig_folder,Rho(i));
    exportgraphics(fig,figure_file,'ContentType','vector');

    clf
    ylabel('sufficiency')
    xlabel('running time in s')
    ylim([0 1e-1]);
    %title(sprintf('Sufficiency scores ($\\rho=%3g$)',Rho(i)),'Interpreter','latex');
    set(gca,'FontName','CMU Serif');
    GAP_stats_i= stats(stats(:,1)==i & stats(:,2)==1,3:end);
    BA_stats_i = stats(stats(:,1)==i & stats(:,2)==0,3:end);
    hold on
    plot(GAP_stats_i(:,1), GAP_stats_i(:,6), '-', 'LineWidth',2,'Color',dblue)
    plot(BA_stats_i(:,1), BA_stats_i(:,6), '-', 'LineWidth',2,'Color',dorange)
    lgd = legend(ours_text,alt_text);
    %title(lgd,'Algorithm')
    lgd.Location = 'north east';
    figure_file = sprintf('%sdynamic_sufficiency_comparison_%1.1f.pdf',fig_folder,Rho(i));
    exportgraphics(fig,figure_file,'ContentType','vector');

    clf
    ylabel('mean attention distance')
    xlabel('running time in s')
    %title(sprintf('Mean attention distance ($\\rho=%3g$)',Rho(i)),'Interpreter','latex');
    set(gca,'FontName','CMU Serif');
    GAP_stats_i= stats(stats(:,1)==i & stats(:,2)==1,3:end);
    BA_stats_i = stats(stats(:,1)==i & stats(:,2)==0,3:end);
    hold on
    plot(GAP_stats_i(:,1), GAP_stats_i(:,8), '-', 'LineWidth',2,'Color',dblue)
    plot(BA_stats_i(:,1), BA_stats_i(:,8), '-', 'LineWidth',2,'Color',dorange)
    lgd = legend(ours_text,alt_text);
    %title(lgd,'Algorithm')
    lgd.Location = 'north east';
    ylim([0 1e-1]);
    figure_file = sprintf('%sdynamic_IE_comparison_%1.1f.pdf',fig_folder,Rho(i));
    exportgraphics(fig,figure_file,'ContentType','vector');
    ylim('auto')

    clf
    ylabel('belief distance')
    xlabel('running time in s')
    %title(sprintf('Maximal belief distance ($\\rho=%3g$)',Rho(i)),'Interpreter','latex');
    set(gca,'FontName','CMU Serif');
    GAP_stats_i= stats(stats(:,1)==i & stats(:,2)==1,3:end);
    BA_stats_i = stats(stats(:,1)==i & stats(:,2)==0,3:end);
    hold on
    plot(GAP_stats_i(:,1), GAP_stats_i(:,9), '-', 'LineWidth',2,'Color',dblue)
    plot(BA_stats_i(:,1), BA_stats_i(:,9), '-', 'LineWidth',2,'Color',dorange)
    lgd = legend(ours_text,alt_text);
    %title(lgd,'Algorithm')
    lgd.Location = 'north east';
    figure_file = sprintf('%sdynamic_belief_comparison_%1.1f.pdf',fig_folder,Rho(i));
    exportgraphics(fig,figure_file,'ContentType','vector');
end
