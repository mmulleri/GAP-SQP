function [stable,fname] = RI_portfolio_step(combo,r,i,p_consider,olddta)
load(sprintf('portfolio_input/%s_setup.mat',combo));
load(sprintf('portfolio_input/%s.mat',olddta));
fname=sprintf('%s_res%02i_rnd%02i',combo,r,i);
diary(sprintf('portfolio_logs/%s.log',fname));
fprintf('Resolution level %2i, round %2i:\n',r,i);
used=(p_marg>max(p_marg)*min_prob);
actiongrid_old=actiongrid(used,:);
p_marg=p_marg(used)/sum(p_marg(used));

fprintf(' - starting with %i actions chosen from a set of %i actions.\n',sum(used),size(actiongrid,1));

xticks=linspace(xrange_FI(1),xrange_FI(2),2^r+1)';
actiongrid_new=generateactiongrid(xticks);

% compute z scores:
psi=prior./b;
zs=zeros(size(actiongrid_new,1),1);
one=min(1,b'*psi); 
%make variables available to parfor workers
stategrid=stategrid; u=u; lambda=lambda; b_logscales=b_logscales;
parfor i=1:size(actiongrid_new,1)
    ui=u(actiongrid_new(i,:),stategrid);
    zsi=exp(ui/lambda + b_logscales)'*psi-one;
    zs(i)=zsi;
end
Delta = max(zs);
delta = -Delta * p_consider/(1-p_consider);
actiongrid_add=actiongrid_new(zs>=delta,:);
z_add=zs(zs>=delta);
fprintf(' - The %2.2f%% cover contains %i actions.\n',p_consider*100,size(actiongrid_add,1));
% exclude actions that are already used in the current solution
[actiongrid_add,keep]=setdiff(actiongrid_add,actiongrid_old,'rows');
z_add=z_add(keep);

stable=false;
if (size(setdiff(actiongrid_add,actiongrid,'rows'),1)==0)
    actiongrid=[actiongrid_old;actiongrid_add];
    p_marg((end+1):size(actiongrid,1),1)=0;
    fprintf(' - All actions in the %2.2f%% cover have already been considered in the last round.\n', p_consider*100);
    stable=true;
else
    [~,usea] = maxk(z_add,K-size(actiongrid_old,1));
    actiongrid=[actiongrid_old;actiongrid_add(usea,:)];
    fprintf(' - Limiting menu to the %i currently used actions and the %i actions with highest score.\n',...
        size(actiongrid_old,1),length(usea));
    u_mat=u(actiongrid,stategrid);
    p_marg((end+1):size(actiongrid,1),1)=0;
    [p_marg,~,~,exitflag,info] = GAP_SQP(u_mat,prior,lambda,'display','iter','actionlabels',actiongrid(:,2:3),'initial_p',p_marg,'zero_tol',0);
    b=info.b_mat*p_marg;
    b_logscales=info.b_logscales;
    GAP_printmarg(p_marg,'actionlabels',actiongrid(:,2:3),'zero_tol',max(p_marg)*min_prob);
end

fig=figure('PaperPosition',[0,0,8,8],'PaperSize',[8,8]);
cla
hold on
scatter(actiongrid(:,2),actiongrid(:,3),3,'r','filled')
scatter(actiongrid(p_marg>0,2),actiongrid(p_marg>0,3),p_marg(p_marg>0)*1000,'k')
hold off
title(sprintf('Marginals over dynamic grid'));
ylim(xrange_FI);
xlim(xrange_FI);
legend('grid','marginals');
print(fig,'-dpdf',sprintf('portfolio_figs/%s.pdf',fname));
close

save(sprintf('portfolio_input/%s.mat',fname),'p_marg','actiongrid','b','b_logscales','stable');
diary off
end