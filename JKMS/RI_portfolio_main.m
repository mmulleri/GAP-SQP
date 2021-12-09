function RI_portfolio_main(combo,ncores)
%% Initialize
maxNumCompThreads(ncores);

addpath('..');

% create required subfolders if they don't yet exist
if ~exist('portfolio_output', 'dir')
    mkdir('portfolio_output')
end
if ~exist('portfolio_input', 'dir')
    mkdir('portfolio_input')
end
if ~exist('portfolio_logs', 'dir')
    mkdir('portfolio_logs')
end
if ~exist('portfolio_figs', 'dir')
    mkdir('portfolio_figs')
end

% setup

switch combo
    case 'A'
        alpha=2;
        lambda=0.05;
        s_z2= 0.0173^2;
    case 'B'
        alpha=1;
        lambda=0.1;
        s_z2= 0.0173^2;
    case 'C'
        alpha=1;
        lambda=0.05;
        s_z2= 0.0173^2;
    case 'D'
        alpha=1;
        lambda=0.05;
        s_z2= 0.03^2;
end
fprintf('Running scenario %s.\n',combo);

generateticks = @(range,N) ((0:1:(N-1))*((range(2)-range(1))/(N-1))+range(1))';

yN=300; % grid resolution
p_marg=[];
actiongrid=zeros(0,3);
s_y2= 0.02^2;
mu_y_risky= 1.04;
mu_y_safe=1.03;
yrange=mu_y_risky+[-1,1]*3*sqrt(s_y2);
yticks=generateticks(yrange,yN);
stategrid=generategrid(yticks,mu_y_safe);
Phi=@(y) normcdf(y,mu_y_risky,sqrt(s_y2));
[~,ppi]=GAP_discretize(@(x,y) 1,1,yticks,Phi);
[a,b]=ndgrid(1:yN,1:yN);
prior= ppi(a,:).*ppi(b,:);
u = @(th,y) (-exp(-alpha*th*y'+alpha^2/2*th.^2*[0;1;1]*s_z2))'; % both given as row vectors

% compute full and no information marginals
xrange_FI=(-mu_y_safe+ yrange)/(alpha*s_z2);
FI_marg=prior;
FI_actions_risky=(stategrid(:,2:3)-stategrid(:,1))/(alpha*s_z2);
FI_actions = [1-sum(FI_actions_risky,2) FI_actions_risky];
w_FI = arrayfun(@(i) u(FI_actions(i,:),stategrid(i,:)),1:size(stategrid,1))*FI_marg;

u_NI = @(y) u([1-2*y,y,y],stategrid)'*prior;
du_NI= @(y) (u([1-2*y,y,y],stategrid).*(3*stategrid(:,1)-sum(stategrid,2)+2*alpha*y*s_z2))'*prior;
yNIticks=(0:.001:1)*(xrange_FI(2)-xrange_FI(1))+xrange_FI(1);
dNI=arrayfun(du_NI,yNIticks);
NI_lb=max(yNIticks(dNI>0));
NI_ub=min(yNIticks(dNI<0));
NI_y=fzero(du_NI,[NI_lb,NI_ub]);
NI_action=[1-2*NI_y,NI_y,NI_y];
NI_marg=1;
w_NI = u(NI_action,stategrid)'*prior;

% start with NI solution
actiongrid=NI_action;
p_marg=[1];
fprintf('Initial estimate from NI solution:\n');
GAP_printmarg(p_marg,'actionlabels',actiongrid);

u_mat=u(actiongrid,stategrid);
b_logscales = -max(u_mat,[],2)/lambda;
b_mat = exp(u_mat/lambda + b_logscales);
b=b_mat*p_marg;

% computation parameters
K=300; % number of actions considered simultaneously
p_consider=0.9999; % consideration set that determines stabilization
start_resolution=3; %initial actiongrid has 2^start_resolution+1 gridpoints in each dimension
final_resolution=9; %final actiongrid has 2^final_resolution+1 gridpoints in each dimension
max_rounds=20; % maximum number of subroutines per resolution level
min_prob=1e-4; % proportionality factor from minimal to maximal marginal
%maxz_tol=1e-15;
save(sprintf('portfolio_input/%s_setup.mat',combo),'u','min_prob','stategrid','xrange_FI','prior','K','lambda','alpha','s_z2','s_y2','mu_y_risky','mu_y_safe');

%% GAP_SQP with iterative grid
fname=sprintf('%s_NI_solution',combo);
save(sprintf('portfolio_input/%s.mat',fname),'p_marg','actiongrid','b','b_logscales');
i=[];
for r=start_resolution:final_resolution
    stable=false;
    i=1;
    while (~stable)
        fprintf('Resolution %i, round %i. \n',r,i);
        fname_new=sprintf('portfolio_input/%s_res%02i_rnd%02i.mat',combo,r,i);
        if ~exist(fname_new, 'file')
            % don't execute if solution already exists
            [stable,fname] = RI_portfolio_step(combo,r,i,p_consider,fname);
        else
            load(fname_new,'stable');
        end

        if (i<max_rounds)
            i=i+1;
        else
            warning('Exceeded number of subroutines set by max_rounds (%i).',max_rounds);
            i=i+1;
            stable=true;
        end
    end
end
load(sprintf('portfolio_input/%s_res%02i_rnd%02i.mat',combo,final_resolution,i-1));
u_mat=u(actiongrid,stategrid);

[~,p_joint,uopt,Iopt,uinfo] = GAP_components(p_marg,u_mat,lambda,prior);


%% comparison
actiongrid_sims=[];p_marg_sims=[];p_joint_sims=[];u_sims=[];I_sims=[];u_mat_sims=[];
if ~exist(sprintf("portfolio_input/%s_sims.mat",combo), 'file')
    RI_portfolio_sims(combo);
end
load(sprintf("portfolio_input/%s_sims.mat",combo));

if ~exist(sprintf("portfolio_input/%s_LQG.mat",combo), 'file')
    RI_portfolio_Gaussian(combo,ncores,100);
end
load(sprintf("portfolio_input/%s_LQG.mat",combo),'u_LQG','mean_ttheta','cov_ttheta','Sig_LQG','I_LQG');

fprintf('w(GAP_SQP) = %5g, w(RIsims) = %5g, w(LQG) = %5g, w(FI) = %5g; w(NI) = %5g.\n',uopt,u_sims,u_LQG,w_FI,w_NI);
fprintf('Payoff difference w(GAP_SQP)-w(RIsims) = %5g, w(GAP_SQP)-w(LQG) = %5g.\n',uopt-u_sims,uopt-u_LQG);
fprintf('JKMS closes %5g of the gap between no and full information.\n',(u_sims-w_NI)/(w_FI-w_NI));
fprintf('LQG closes %5g of the gap between no and full information.\n',(u_LQG-w_NI)/(w_FI-w_NI));
fprintf('GAP_SQP closes %5g of the gap between no and full information.\n',(uopt-w_NI)/(w_FI-w_NI));

save(sprintf('portfolio_output/%s_final.mat',combo),'stategrid','u','lambda','prior','b_logscales','actiongrid','actiongrid_sims','u_mat','u_mat_sims','Sig_LQG','I_LQG','b','alpha','p_marg','p_marg_sims','min_prob','xrange_FI','r','i','mean_ttheta','cov_ttheta','p_joint','Iopt','p_joint_sims','I_sims','NI_action');

RI_portfolio_plots(combo,ncores);

%%% export results for sims code
% x=actiongrid(p_marg>0,2:3);
% px=p_marg(p_marg>0);
% px=px/sum(px);
% nx=length(px);
% ygivenx=p_joint(:,p_marg>0)';
% ygivenx=ygivenx./sum(ygivenx,2);
% ygivenx_sims=p_joint_sims'./sum(p_joint_sims',2);
% x_sims=actiongrid_sims(:,2:3);
% nx_sims=size(x_sims,1);
% px_sims=p_marg_sims;
% y=stategrid(:,2:3);
% gy=prior;
% save(sprintf('portfolio_solutions/portfolio_%s_JKMS_export.mat',combo),'uopt','u_sims','x','px','nx','ygivenx','ygivenx_sims','nx_sims','px_sims','x_sims','y','gy','uinfo','uinfo_sims');
end
