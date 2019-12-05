function RI_portfolio_plots(combo,yN,ncores)
RI_portfolio_init
if (ncores>1) 
    parpool('local', ncores);
end
stategrid=stategrid; % this makes the variable accessible to the parfor workers
u=u;
lambda=lambda;
b_logscales=b_logscales;

% compute z scores:
        b=b_mat*p_marg;
        psi=prior./b;
r=9;
[l1 u1]= bounds(actiongrid(:,2));
[l2 u2]= bounds(actiongrid(:,3));
lb=min(l1,l2); ub=min(u1,u2);
plotrange=1.2*[lb ub];
xticks=linspace(xrange_FI(1),xrange_FI(2),2^r+1)';
xticks=xticks(xticks>=plotrange(1) & xticks<=plotrange(2));
actiongrid_new=generateactiongrid(xticks);
actiongrid_new=setdiff(actiongrid_new,actiongrid,'rows');
actiongrid=[actiongrid;actiongrid_new];
p_marg=[p_marg;zeros(size(actiongrid,1)-length(p_marg),1)];

fprintf('Now finding partial covers.\n');
zs=zeros(size(actiongrid,1),1);
one=min(1,b'*psi);
parfor i=1:size(actiongrid,1)
            ui=u(actiongrid(i,:),stategrid);
            zsi=exp(ui/lambda + b_logscales)'*psi-one;
            zs(i)=zsi;
end
Delta = max(zs);
p_cover=[0.999,.99,.95];
delta = -Delta * p_cover./(1-p_cover);
covers=(zs>=delta);
save(sprintf('portfolio_solutions/portfolio_%s_%i_zscores.mat',combo,yN),'zs','actiongrid','p_marg','covers','p_cover');

fig=figure('PaperPosition',[0,0,3,3],'PaperSize',[3,3]);
gray=0.8;
clf
hold on
scatter(actiongrid(covers(:,1)==1,2),actiongrid(covers(:,1)==1,3),5,gray*[1 1 1],'filled')
scatter(actiongrid(covers(:,3)==1,2),actiongrid(covers(:,3)==1,3),5,'k','filled')
hold off
ylim(plotrange);
xlim(plotrange);
xlabel('\theta_1');
ylabel('\theta_2');
legend('99.9% cover','95.0% cover','Location',[0.5 0.5 0 0]);
print(fig,'-dpdf',sprintf('portfolio_figs/%s_final_bounds_%i.pdf',combo,yN));
close

fig=figure('PaperPosition',[0,0,3,3],'PaperSize',[3,3]);
clf
hold on
scatter(actiongrid_sims(:,2),actiongrid_sims(:,3),p_marg_sims*1000,'b')
scatter(actiongrid(p_marg>0,2),actiongrid(p_marg>0,3),p_marg(p_marg>0)*1000,'k')
hold off
ylim(plotrange);
xlim(plotrange);
xlabel('\theta_1');
ylabel('\theta_2');
legend('JKMS','GAP-SQP','Location',[0.5 0.5 0 0]);
print(fig,'-dpdf',sprintf('portfolio_figs/%s_final_marginals2D_%i.pdf',combo,yN));
close

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
