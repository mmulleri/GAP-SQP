function RI_portfolio_plots(combo,ncores)
maxNumCompThreads(ncores);
load(sprintf('portfolio_output/%s_final.mat',combo),'stategrid','u','lambda','prior','b_logscales','actiongrid','actiongrid_sims','u_mat','u_mat_sims','b','alpha','p_marg','p_marg_sims','min_prob','xrange_FI','r','i','mean_ttheta','cov_ttheta','Sig_LQG','I_LQG','p_joint','Iopt','p_joint_sims','I_sims','NI_action');
load(sprintf('portfolio_input/%s_setup.mat',combo),'mu_y_safe','mu_y_risky');

%% compute z scores:
psi=prior./b;
keep=p_marg>max(p_marg)*min_prob;
[l1 u1]= bounds(actiongrid(keep,2));
[l2 u2]= bounds(actiongrid(keep,3));
lb=min(l1,l2); ub=min(u1,u2);
plotrange=[lb-5 ub+5];
xticks=linspace(xrange_FI(1),xrange_FI(2),2^r+1)';
xticks=xticks(xticks>=plotrange(1) & xticks<=plotrange(2));
actiongrid_new=generateactiongrid(xticks);
actiongrid_new=setdiff(actiongrid_new,actiongrid,'rows');
actiongrid_full=[actiongrid;actiongrid_new];
p_marg_plot=[p_marg;zeros(size(actiongrid,1)-length(p_marg),1)];

if ~exist(sprintf("portfolio_input/%s_zscores.mat",combo), 'file')
    fprintf('Now finding partial covers.\n');
    zs=[];
    one=min(1,b'*psi);
    parfor i=1:size(actiongrid_full,1)
        ui=u(actiongrid_full(i,:),stategrid);
        zsi=exp(ui/lambda + b_logscales)'*psi-one;
        zs(i)=zsi;
    end
    Delta = max(zs);
    p_cover=[0.9999,.99,.95];
    delta = -Delta * p_cover./(1-p_cover);
    covers=(zs'>=delta);
    save(sprintf('portfolio_input/%s_zscores.mat',combo),'zs','plotrange','actiongrid_full','p_marg','p_marg_plot','covers','p_cover','combo');
else
    load(sprintf('portfolio_input/%s_zscores.mat',combo))
end

%% Figures
dblue='#143D73';
lblue='#99B1C3';
llblue='#C0D0DC';
dorange='#F29F05';
dred='#BF214B';
gray='#808080';

fig=figure(1);
fig.Units = 'inches';
plotsize=[3 3];
fig.Position = [1 2 plotsize];
ylim(plotrange); set(gca,'ytick',-40:20:60);
xlim(plotrange); set(gca,'xtick',-40:20:60);
set(gca,'FontSize',10);
set(gca,'FontName','CMU Serif');
xlabel('\theta_1');
ylabel('\theta_2');

%% Bounds
cla
hold on
scatter(actiongrid_full(covers(:,1)==1,2),actiongrid_full(covers(:,1)==1,3),10,'MarkerEdgeColor',llblue,'MarkerFaceColor',llblue)
figure_file = sprintf('portfolio_figs/%s_final_bounds_9999.pdf',combo);
scatter(actiongrid_full(covers(:,3)==1,2),actiongrid_full(covers(:,3)==1,3),10,'MarkerEdgeColor',dblue,'MarkerFaceColor',dblue)
hold off
lgd=legend('99.99% cover','95% cover','Location','northoutside');
lgd.Box='off';lgd.NumColumns=2;lgd.Units='inches';
fig.Position=[fig.Position(1:2),plotsize(1),plotsize(2)+lgd.Position(4)];

figure_file = sprintf('portfolio_figs/%s_final_bounds.pdf',combo);
exportgraphics(fig,figure_file,'ContentType','vector');

%% Estimates
cla
hold on

scatter(NI_action(2),NI_action(3),1000,'MarkerEdgeColor',dred,'MarkerFaceColor',dred,'MarkerFaceAlpha',0.3,'LineWidth',1);

LQGticks=linspace(plotrange(1),plotrange(2),100);
[X1,X2] = meshgrid(LQGticks',LQGticks');
p = reshape(mvnpdf([X1(:) X2(:)],mean_ttheta',cov_ttheta),100,100);
lvls=zeros(0,1);
for plevel=0.2:0.2:.8
    % find point at specific mahalonobis distance (for details, see
    % https://upload.wikimedia.org/wikipedia/commons/a/a2/Cumulative_function_n_dimensional_Gaussians_12.2013.pdf)
    pt=fzero(@(x) [x-mean_ttheta(1);0]'*(cov_ttheta)^(-1)*[x-mean_ttheta(1);0]+2*log(1-plevel),mean_ttheta(2));
    lvls(end+1)=mvnpdf([pt;mean_ttheta(2)],mean_ttheta,cov_ttheta);
end
contour(LQGticks,LQGticks,p,lvls,'Color',gray,'Linewidth',0.5);
scatter(mean_ttheta(1),mean_ttheta(2),1,'MarkerEdgeColor',gray,'MarkerFaceColor',gray);

s=scatter(actiongrid_sims(:,2),actiongrid_sims(:,3),p_marg_sims*1000-2*ones(length(p_marg_sims),1),'MarkerEdgeColor',dorange,'MarkerFaceColor',dorange,'MarkerFaceAlpha',0.3,'LineWidth',1);
%to plot lines inside marker, need to distinguish between 'big enough' and
%'small' likelihood choices
likely=p_marg>2*1e-3;
unlikely=p_marg>0 & p_marg<=2*1e-3;
s=scatter(actiongrid(likely,2),actiongrid(likely,3),p_marg(likely)*1000-2*ones(sum(likely),1),'MarkerEdgeColor',dblue,'MarkerFaceColor',dblue,'MarkerFaceAlpha',0.3,'LineWidth',1);
scatter(actiongrid(unlikely,2),actiongrid(unlikely,3),p_marg(unlikely)*1000,'MarkerFaceColor',dblue,'MarkerEdgeColor','none');

hold off

lgd=legend('no information','Gaussian','','JKMS','GAP-SQP');
lgd.Location = 'northoutside'; lgd.Box='off'; lgd.NumColumns=2;
lgd.Units='inches';
fig.Position=[fig.Position(1:2),plotsize(1),plotsize(2)+lgd.Position(4)];

figure_file = sprintf('portfolio_figs/%s_final_marginals2D.pdf',combo);
exportgraphics(fig,figure_file,'ContentType','vector');

%% Compute Ignorance Equivalent
grid=linspace(-.8,0,200);
%weighted_hist = @(w,x) arrayfun(@(ub) sum(w(x<ub),'all'),grid(2:end)) - arrayfun(@(lb) sum(w(x<lb),'all'),grid(1:end-1));
bw=0.01; %bandwidth

u_FI=zeros(size(stategrid,1),1);
parfor (k=1:size(stategrid,1))
    u_FI(k)=max(u(actiongrid_full,stategrid(k,:)));
end
%FIhist=weighted_hist(prior,u_FI);
%NIhist=weighted_hist(prior,u(NI_action,stategrid));
[pdfFI,uFI]=ksdensity(u_FI,grid,'Weights',prior,'Bandwidth',bw);
[pdfNI,uNI]=ksdensity(u(NI_action,stategrid),grid,'Weights',prior,'Bandwidth',bw);

weightsGAP=p_joint; IGAP=Iopt;
%GAPhist=weighted_hist(weightsGAP(:),u_mat(:)-lambda*IGAP);
[pdfGAP,uGAP]=ksdensity(u_mat(:)-lambda*IGAP,grid,'Weights',weightsGAP(:),'Bandwidth',bw);
weightsJKMS=p_joint_sims; IJKMS=I_sims;
%JKMShist=weighted_hist(weightsJKMS(:),u_mat_sims(:)-lambda*IJKMS);
[pdfJKMS,uJKMS]=ksdensity(u_mat_sims(:)-lambda*IJKMS,grid,'Weights',weightsJKMS(:),'Bandwidth',bw);
%Monte Carlo simulation of Gaussian utility:
U_obj = @(ttheta1,ttheta2,y1,y2) u([1-ttheta1-ttheta2, ttheta1, ttheta2],[mu_y_safe mu_y_risky+y1 mu_y_risky+y2]);
LQGdraws = mvnrnd([mean_ttheta;0;0]',Sig_LQG,1e8);
u_list_LQG = arrayfun(U_obj,LQGdraws(:,1),LQGdraws(:,2),LQGdraws(:,3),LQGdraws(:,4));
[pdfLQG,uLQG]=ksdensity(u_list_LQG- lambda*I_LQG,grid,'Bandwidth',bw);

IE = @(p,b_mat) lambda*(log(b_mat*p)-b_logscales);
b_logscales = -max(u_mat,[],2)/lambda;
b_mat = exp(u_mat/lambda + b_logscales);
[pdfIE,uIE]=ksdensity(IE(p_marg,b_mat),grid,'Weights',prior,'Bandwidth',bw);
%IEhist=weighted_hist(prior,IE(p_marg,b_mat));

% IE plot
fig.Position = [1 2 7 3];
cla 
hold on
plot(uGAP,pdfGAP,'Color',dblue,'Linewidth',2);
%stairs(grid,[GAPhist,0],'Color',dblue,'LineWidth',2);
plot(uFI,pdfFI,'k--','Linewidth',1);
plot(uNI,pdfNI,'k:','Linewidth',1);
plot(uIE,pdfIE,'Color',dred,'Linewidth',2);
hold off
expct=weightsGAP(:)'*u_mat(:)-lambda*IGAP;
%xline(expct);
legend('GAP-SQP','free information','no information','ignorance equivalent','Location','northwest');
ylim('auto'); set(gca,'ytick',0:1:10);
xlim([-.8 0]); set(gca,'ytick',-1:.1:0);
xlabel('utility level');
ylabel('density');

figure_file = sprintf('portfolio_figs/%s_final_payoff_pdf_300.pdf',combo);
exportgraphics(fig,figure_file,'ContentType','vector');

%% ACROSS ALGORITHMS
cla
hold on
%stairs(grid,[GAPhist,0],'Color',dblue,'LineWidth',1);
%stairs(grid,[JKMShist,0],'Color',dorange,'LineWidth',1);
plot(uLQG,pdfLQG,'Color',gray,'Linewidth',2);
plot(uJKMS,pdfJKMS,'Color',dorange,'Linewidth',2);
plot(uGAP,pdfGAP,'Color',dblue,'Linewidth',2);
hold off
legend('Gaussian','JKMS','GAP-SQP','Location','northwest');

figure_file = sprintf('portfolio_figs/%s_final_payoff_pdf_algos_300.pdf',combo);
exportgraphics(fig,figure_file,'ContentType','vector');

%% wrap up
close
end

