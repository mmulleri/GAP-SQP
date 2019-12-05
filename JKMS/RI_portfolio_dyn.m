function RI_portfolio_dyn(combo,yN,ncores)
RI_portfolio_init
mypool=parpool('local', ncores);
%% grant access to relevant variables
stategrid=stategrid; 
u=u;
lambda=lambda;
b_logscales=b_logscales;
b_mat=b_mat;
u_mat=u_mat;

%% computation parameters
K=10000; % number of actions considered simultaneously
p_consider=0.99; % consideration set that determines stabilization
start_resolution=10; %initial actiongrid has 2^start_resolution+1 gridpoints in each dimension
final_resolution=10; %final actiongrid has 2^final_resolution+1 gridpoints in each dimension
max_rounds=10; % maximum number of subroutines per resolution level
min_prob=1e-4; % proportionality factor from minimal to maximal marginal
maxz_tol=1e-15;

%% RI solve with iterative grid
actiongrid_untruncated=actiongrid;

% dominated_actions = zeros(0,3);
for r=start_resolution:final_resolution
    filename=sprintf('portfolio_output/%s_log_%02i.out',combo,r);
    diary(filename);
    i=1;
    stable=false;
    while (~stable)
        actiongrid_old=actiongrid;
        psi=prior./(b_mat*p_marg);
        z = @(a) psi'*exp(u(a,stategrid)/lambda + b_logscales)-1;
        
        xticks=linspace(xrange_FI(1),xrange_FI(2),2^r+1)';
        actiongrid_new=generateactiongrid(xticks);
%         % consider only actions that are not dominated
%         actiongrid_new=setdiff(actiongrid_new,dominated_actions,'rows');
        % consider only actions that are not already in the old grid
        actiongrid_new=setdiff(actiongrid_new,actiongrid_old,'rows');
        
        % compute z scores:
        b=b_mat*p_marg;
        psi=prior./b;
        zs=zeros(size(actiongrid_new,1),1);
        one=min(1,b'*psi);
        parfor i=1:size(actiongrid_new,1)
            ui=u(actiongrid_new(i,:),stategrid);
            zsi=exp(ui/lambda + b_logscales)'*psi-one;
            zs(i)=zsi;
        end
        Delta = max(zs);
        delta = -Delta * p_consider/(1-p_consider);
        actiongrid_add=actiongrid_new(zs>=delta,:);
        z_add=zs(zs>=delta);
        
        if (size(setdiff(actiongrid_add,actiongrid_untruncated,'rows'),1)==0)
            fprintf('Resolution level %2i, round %2i:\n - Currently using %i actions.\n - Full grid adds %i actions.\n - Total cardinality of %2.2f%% cover is %i.\n - No additional actions are within the %2.2f%% cover.\n',...
                r,i,size(actiongrid_old,1),size(actiongrid_new,1),p_consider*100,size(actiongrid_add,1)+size(actiongrid_old,1),p_consider*100);
            actiongrid=actiongrid_old;
            stable=true;
            continue
        end
        
        [~,usea] = maxk(z_add,K-size(actiongrid_old,1));
        actiongrid=[actiongrid_old;actiongrid_add(usea,:)];
        fprintf('Resolution level %2i, round %2i:\n - Currently using %i actions.\n - Full grid adds %i actions.\n - Total cardinality of %2.2f%% cover is %i.\n - Limiting menu to the %i existing actions and the %i actions with highest score.\n',...
            r,i,size(actiongrid_old,1),size(actiongrid_new,1),p_consider*100,size(actiongrid_add,1)+size(actiongrid_old,1),size(actiongrid_old,1),length(usea));
        u_mat=u(actiongrid,stategrid);
        p_marg((end+1):size(actiongrid,1),1)=0;
        [p_marg,~,~,exitflag,info] = GAP_SQP(u_mat,prior,lambda,'display','iter','actionlabels',actiongrid(:,2:3),'initial_p',p_marg);
        b_mat=info.b_mat;
        b_logscales=info.b_logscales;
%         % determine if any actions are interior to the convex hull spanned
%         % by b_mat
%         I = size(b_mat,1);
%         J = size(b_mat,2);
%         psi=optimvar('psi',I,'LowerBound',0);
%         prob_psi = optimproblem();
%         prob_psi.Constraints.ineq = (b_mat'*psi <= ones(J,1));
%         Jtot = size(actiongrid_new,1);
%         Z0 = zeros(Jtot,1);
%         parfor j=1:Jtot
%             prob_psi_j=prob_psi;
%             uj = u(actiongrid_new(j,:),stategrid);
%             bj = exp(uj/lambda + b_logscales);
%             prob_psi_j.Objective= -bj'*psi;
%             maxzj = prob2struct(prob_psi_j);
%             maxzj.options=optimset('Display',"none");
%             [psizj,zj,found] = linprog(maxzj);
%             if found==1
%                 Z0(j) = -zj-1;
%             end
%         end
%         dominated = (Z0<-1e-12);          
%         fprintf(' - %i actions are interior to the learning-proof menu.\n',sum(dominated));
%         dominated_actions = [dominated_actions; actiongrid_new(dominated,:)];
        
        % drop actions with small probability
        keep = any(p_marg>min_prob*max(p_marg),2);
        plotgrid=actiongrid;
        actiongrid_untruncated=actiongrid;
        actiongrid(~keep,:)=[];
        b_mat(:,~keep)=[];
        u_mat(:,~keep)=[];
        p_marg(~keep)=[]; p_marg = p_marg/sum(p_marg); %trim marginals
        GAP_printmarg(p_marg,'actionlabels',actiongrid(:,2:3));
        
        save(sprintf('portfolio_solutions/portfolio_%s_Y%i_it%02i%02i.mat',combo,yN,r,i),'p_marg','actiongrid','stategrid','prior');
        
        fig=figure('PaperPosition',[0,0,8,8],'PaperSize',[8,8]);
        clf
        scatter(plotgrid(:,2),plotgrid(:,3),3,'r','filled')
        hold on
        scatter(actiongrid(:,2),actiongrid(:,3),p_marg*1000,'k')
        hold off
        title(sprintf('Marginals over dynamic grid'));
        ylim(xrange_FI);
        xlim(xrange_FI);
        legend('RIsolve grid','RIsolve marginals');
        print(fig,'-dpdf',sprintf('portfolio_figs/%s_marginals_yN%i_it%02i%02i.pdf',combo,yN,r,i));
        close
        
        i=i+1;
        if i>max_rounds
            fprintf('Resolution level %2i reached maximum number of subroutines before the %2.2f%% cover stabilized.\n',...
                r,p_consider*100);
            stable=true;
        end
    end
    diary off
end

[~,p_joint,uopt,Iopt,uinfo] = GAP_components(p_marg,b_mat,u_mat,lambda,prior);

%% sims call
actiongrid_sims(:,1)=1-actiongrid_sims(:,2)-actiongrid_sims(:,3);
p_marg_sims=p_marg_sims/sum(p_marg_sims);
u_mat_sims   = u(actiongrid_sims,stategrid);
b_mat_sims   = exp(u_mat_sims/lambda+b_logscales);
b_sims       = b_mat_sims*p_marg_sims;
[~,p_joint_sims,u_sims,I_sims,uinfo_sims] = GAP_components(p_marg_sims,b_mat_sims,u_mat_sims,lambda,prior);


%% comparison
fprintf('Payoff difference w(RIquad)-w(RIsims) = %5g.\n',uopt-u_sims);

grand_u = [u_mat u_mat_sims];
grand_b = [b_mat b_mat_sims];
grand_sims = [zeros(length(p_marg),1);p_marg_sims];
grand_p = [p_marg;zeros(length(p_marg_sims),1)];
[~,~,~,~,uinfo_sims] = GAP_components(grand_sims,grand_b,grand_u,lambda,prior);
fprintf('JKMS closes %5g of the gap between no and full information.\n',uinfo_sims.normalized);
[~,~,~,~,uinfo] = GAP_components(grand_p,grand_b,grand_u,lambda,prior);
fprintf('RIquad closes %5g of the gap between no and full information.\n',uinfo.normalized);

% fprintf('We should have the following equal:\n');
% [u_NId,I]=max(prior'*u_mat);
% [u_NId,u_NI(NI_y)]
% [actiongrid(I,:);NI_action]

save(sprintf('portfolio_solutions/portfolio_%s_%i_final.mat',combo,yN));

%% export results for sims code
x=actiongrid(p_marg>0,2:3);
px=p_marg(p_marg>0);
px=px/sum(px);
nx=length(px);
ygivenx=p_joint(:,p_marg>0)';
ygivenx=ygivenx./sum(ygivenx,2);
ygivenx_sims=p_joint_sims'./sum(p_joint_sims',2);
x_sims=actiongrid_sims(:,2:3);
nx_sims=size(x_sims,1);
px_sims=p_marg_sims;
y=stategrid(:,2:3);
gy=prior;
save(sprintf('portfolio_solutions/portfolio_%s_%i_JKMS_export.mat',combo,yN),'uopt','u_sims','x','px','nx','ygivenx','ygivenx_sims','nx_sims','px_sims','x_sims','y','gy','uinfo','uinfo_sims');
end

function [grid]=generategrid(ticks,dflt)
N = length(ticks);
[a,b]=ndgrid(1:N,1:N);
grid=[dflt*ones(N*N,1),ticks(a,:),ticks(b,:)];
end

function grid=generateactiongrid(ticks)
grid=generategrid(ticks,1);
grid(:,1)=1-grid(:,2)-grid(:,3);
end
