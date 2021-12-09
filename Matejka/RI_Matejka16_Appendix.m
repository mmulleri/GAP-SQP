% RI_Matejka16_Appendix.m
%-------------------------------------
% Replicate Matejka(2016) AMPL output for Appendix A.2.
%-------------------------------------

clear;

%% Load and format Matejka's AMPL output

load('AMPL_output.mat')

% Setup
stategrid = X;
actiongrid = Y;
Nstates = length(stategrid);
Nactions = length(actiongrid);

% Solution
pj_ampl=reshape(A,[Nstates Nactions]);
p_ampl=sum(pj_ampl,1)';


%% Parameters

% Constant settings - Model
ufun = @(p,d) p^(-(d+1)/d)*(p-1);

% Information cost
llambda = 0.00531;
% Set to replicate 0.5 bits

% Prior
ppi = ones(Nstates,1)/Nstates;

% Convergence tolerance for IE
conv_tol_IE = 1e-14;

% Figures
set(groot,'DefaultAxesFontSize',14)

%% Setup

u_mat = zeros(Nstates,Nactions);
for s=1:Nstates
    for x=1:Nactions
        u_mat(s,x) = ufun(actiongrid(x),stategrid(s));
    end
end

b_mat = exp(u_mat/llambda);

% Negative of the objective function
neg_w = @(p) -llambda*ppi'*log(b_mat*p);

% Ignorance equivalent
IE = @(p) llambda*log(b_mat*p);

% Conditional from joint
q = @(pjoint) pjoint./(sum(pjoint,1).*sum(pjoint,2));

% Mutual information (in bits)
I = @(p,q) sum(p(p>0).*log(q(p>0)))/log(2);

%% Benchmarks

% Full information
[u_FI,FI_actions] = max(u_mat,[],2);
FI_pjoint = zeros(Nstates,Nactions);
FI_pjoint(sub2ind([Nstates Nactions],1:Nstates,FI_actions'))=1;
p_FI = FI_pjoint'*ppi;
w_FI = sum(ppi'*(FI_pjoint.*u_mat));

% No information
[w_NI,NI_action] = max(ppi'*u_mat);

% Information value
info_value = w_FI - w_NI;
fprintf('Tolerance set to %g, equal to %g percent of information value.\n',...
    conv_tol_IE, conv_tol_IE/info_value*100)


% Values from AMPL output
u = @(p_joint) sum(sum(p_joint.*u_mat))-llambda*I(p_joint,q(p_joint));
w_M = u(pj_ampl);
I_M = I(pj_ampl, q(pj_ampl));


%% SQP algorithm

[p_marg,~,ttime,exitflag] = GAP_SQP(u_mat,ppi,llambda,...
    'display','off',...
    'conv_tol_IE',conv_tol_IE,...
    'save_hist', false);
b = b_mat*p_marg;
p_marg = max(p_marg,0);
p_marg = p_marg/sum(p_marg);

fprintf('GAP-SQP algorithm total running time: %g seconds. \n',ttime)
fprintf('GAP-SQP exitflag: %i.\n', exitflag)

% Retrieve
temp_cond= b_mat*diag(p_marg);
temp_cond=temp_cond./sum(temp_cond,2);
p_joint = diag(ppi)*temp_cond;

Iopt = I(p_joint,q(p_joint));
w_QP = - neg_w(p_marg);

fprintf('Optimal information flow: %g bits. \n', Iopt)

%% Blahut-Arimoto

[p_marg_BA,hist,ttime_BA,exitflag] = solve_BA(u_mat,ppi,llambda,...
    'display','off',...
    'stopping_rule','objective',...
    'save_hist', false);

b_BA = b_mat*p_marg_BA;
p_marg_BA = max(p_marg_BA,0);
p_marg_BA = p_marg_BA/sum(p_marg_BA);
w_BA = - neg_w(p_marg_BA);

fprintf('Blahut-Arimoto algorithm running time: %g seconds. \n',ttime_BA)
fprintf('Blahut-Arimoto exitflag: %i.\n', exitflag)

fprintf('Objective function difference (GAP-SQP - Blahut Arimoto): %g percent of information value. \n',...
    (w_QP - w_BA)/info_value*100)

%% Figures

% Printing parameters
offset = .003;width = 1;

% Compare with SQL
dblue='#143D73';
lblue='#96AFC2';
dorange='#F29F05';
dred='#BF214B';

fig=figure(1);
fig.Units = 'inches';
fig.Position = [0 0 7 2];
hold on



bar(actiongrid,p_marg*100,width,'FaceColor',dblue,'EdgeAlpha', 0)
bar(Y+offset,p_ampl*100,width,'FaceColor',dred,'EdgeAlpha', 0)
hold off
legend('GAP-SQP','AMPL',...
    'Location','NorthWest')
ylabel('probability (percent)')
xlabel('price')

set(gca, 'YGrid', 'on', 'XGrid', 'off')
set(gca,'FontSize',9);
set(gca,'FontName','CMU Serif');
figure_file = 'Matejka16_figs/replicate_SQP.pdf';
exportgraphics(fig,figure_file,'ContentType','vector');

% Compare with Blahut-Arimoto
cla
hold on
ba   = bar(actiongrid,p_marg_BA*100,width,'FaceColor',dorange,'EdgeAlpha', 0);
ba.DisplayName = 'Blahut-Arimotho';
ampl = bar(Y+offset,p_ampl*100,width,'FaceColor',dred,'EdgeAlpha', 0)
ampl.DisplayName = 'AMPL';
hold off
legend([ba ampl],'Location','NorthWest')

figure_file = 'Matejka16_figs/replicate_BA.pdf';
exportgraphics(fig,figure_file,'ContentType','vector');
close

% eof