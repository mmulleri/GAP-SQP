% generate_probsets.m
%-------------------------------------
% Generates probabilistic sets.
%-------------------------------------

clear;

%% Parameters (Benchmark)

% Constant settings - Model
ufun = @(p,d) p^(-(d+1)/d)*(p-1);
staterange = [1/9 1/2];
actionrange = [10/9 3/2];

% Grids
Nstates = 100;
Nactions = 100;
stategrid = transpose(staterange(1):(staterange(2)-staterange(1))/(Nstates-1):staterange(2));
actiongrid = transpose(actionrange(1):(actionrange(2)-actionrange(1))/(Nactions-1):actionrange(2));

% Prior
ppi = ones(Nstates,1)/Nstates;

% Information cost
llambda = 0.00531;

% Convergence tolerance for IE
conv_tol_IE = 1e-11;

% Destination folder
d_folder = 'output\';

%% Setup

u_mat = zeros(Nstates,Nactions);
for s=1:Nstates
    for x=1:Nactions
        u_mat(s,x) = ufun(actiongrid(x),stategrid(s));
    end
end
IE = @(p) llambda*log(b_mat*p);

%% Generate sets
[p_marg,hist,ttime,exitflag,info] = GAP_SQP(u_mat,ppi,llambda,...
    'display','off',...
    'conv_tol_IE',conv_tol_IE); 

covers = GAP_bounds(info.b_mat,ppi,p_marg,[0,.01]);

%% Figure dominated actions
set(groot,'DefaultAxesFontSize',14)


width = 1;
RGB_color = [0.3010 0.7450 0.9330];

figure('PaperPosition',[0,0,9,3],'PaperSize',[9,3])
hold on

bar(actiongrid, (1-covers(:,1))*.6, width,...
    'FaceColor',RGB_color,...
    'EdgeColor',RGB_color,...
    'FaceAlpha', .8, 'EdgeAlpha', 0)
bar(actiongrid,p_marg,width)
lgd = legend('undominated','Numerical solution',...
    'Location','NorthEast');
%title(lgd,'Consideration sets')

ylabel('Probability')
xlabel('Price')
print('-f1','-dpdf','output/replicate_dominate.pdf')
close