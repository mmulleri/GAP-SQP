maxNumCompThreads(ncores);

%% setup

switch combo
    case 'A'
alpha=2;
lambda=0.05;
s_z2= 0.0173^2;
actiongrid_sims=[0, 20.727, 20.727; 0,14.418, 1.429; 0,1.429, 14.418];
p_marg_sims=[0.066;0.467;0.467]; 
    case 'B'
    alpha=1;
    lambda=0.1;
s_z2= 0.0173^2;
actiongrid_sims=[0, -23.006, -23.006; 0,55.214, 10.431; 0,10.431, 55.214];
p_marg_sims=[0.185;0.408;0.408];
    case 'C'
alpha=1;
lambda=0.05;
s_z2= 0.0173^2;
actiongrid_sims=[0, -77.975, -77.975; 0,45.748, -0.091; 0,-0.091, 45.748];
p_marg_sims=[0.024;0.488;0.488]; 
    case 'D'
alpha=1;
lambda=0.05;
s_z2= 0.03^2;
actiongrid_sims=[0,19.328,19.328;0,19.890,-5.666;0,-5.666,19.890;0,-8.737,-8.737];
p_marg_sims=[0.361;0.248;0.248;0.142]; 
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

%% compute full and no information marginals
xrange_FI=(-mu_y_safe+ yrange)/(alpha*s_z2);
FI_marg=prior;
FI_actions=[1-sum(stategrid,2)/(alpha*s_z2),...
    (stategrid(:,2:3)-stategrid(:,1))/(alpha*s_z2)];
u_NI = @(y) u([1-2*y,y,y],stategrid)'*prior;
du_NI= @(y) (u([1-2*y,y,y],stategrid).*(3*stategrid(:,1)-sum(stategrid,2)+2*alpha*y*s_z2))'*prior;

yNIticks=(0:.001:1)*(xrange_FI(2)-xrange_FI(1))+xrange_FI(1);
dNI=arrayfun(du_NI,yNIticks);
NI_lb=max(yNIticks(dNI>0));
NI_ub=min(yNIticks(dNI<0));
NI_y=fzero(du_NI,[NI_lb,NI_ub]);
NI_action=[1-2*NI_y,NI_y,NI_y];
NI_marg=1;

%% sims call
actiongrid_sims(:,1)=1-actiongrid_sims(:,2)-actiongrid_sims(:,3);
p_marg_sims=p_marg_sims/sum(p_marg_sims);
u_mat_sims   = u(actiongrid_sims,stategrid);
b_logscales_sims = -max(u_mat_sims,[],2)/lambda;
b_mat_sims = exp(u_mat_sims/lambda + b_logscales_sims);
b_sims       = b_mat_sims*p_marg_sims;
[~,p_joint_sims,u_sims,I_sims,uinfo_sims] = GAP_components(p_marg_sims,b_mat_sims,u_mat_sims,lambda,prior);

%% import existing solution
importfile=sprintf('portfolio_input/portfolio_%s_Y%i.mat',combo,yN);
if isfile(importfile)
    load(importfile,'actiongrid');
    load(importfile,'p_marg');
    fprintf('Initial estimates from %s:\n',importfile);
    GAP_printmarg(p_marg,'actionlabels',actiongrid);
else
    % start with NI solution
    actiongrid=NI_action;
    p_marg=[1];
    fprintf('Initial estimates from NI solution:\n');
    GAP_printmarg(p_marg,'actionlabels',actiongrid);
end
u_mat=u(actiongrid,stategrid);
b_logscales = -max(u_mat,[],2)/lambda;
b_mat = exp(u_mat/lambda + b_logscales);

function [grid]=generategrid(ticks,dflt)
N = length(ticks);
[a,b]=ndgrid(1:N,1:N);
grid=[dflt*ones(N*N,1),ticks(a,:),ticks(b,:)];
end

function grid=generateactiongrid(ticks)
grid=generategrid(ticks,1);
grid(:,1)=1-grid(:,2)-grid(:,3);
end