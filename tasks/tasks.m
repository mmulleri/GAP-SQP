%% comparative statics
clc
clf
d_folder='tasks_output/';
addpath('..');

N=8; llambdas=10.^(-1.1:.001:2); %=10.^(-1.1:.001:2); 
p = 0.5; %probability that task 1 is critical
q = 1/N; %probability that worker 1 is unskilled
ddelta = 0.95;
wage=0.4;
q = [q; (1-q)/(N-1)*ones(N-1,1)];

%initial solution
[prob,ttime,exitflag,info,setup]=taskAssignment(N,p,q,ddelta,wage,llambdas(1));

for (k=2:length(llambdas))
    fprintf('Information cost = %f.\n',llambdas(k));
    
    [prob,ttime,exitflag,info_add]=taskAssignment(N,p,q,ddelta,wage,llambdas(k),'setup',setup,'initial_guess',prob.marg);
    
    info(end+1)=info_add;
    
end

% Save output
save([d_folder 'tasks_info.mat'])

%% Main figure
dblue='#143D73';
lblue='#96AFC2';
dorange='#F29F05';
dred='#BF214B';

lwd=1.5;
fig=figure(1);
clf
hires=[info.Nhired];
hires_up = [info.Nunprod];
sametask = [info.Nsametask];
semilogx(llambdas,hires(2,:),'-','LineWidth',lwd,'color',dblue);
%title('Strategy');
%subtitle(sprintf('N = %i, delta = %1.2f, p = %1.2f, wage = %1.2f',N,ddelta,p,wage));
ylim([0,N]); xlim([min(llambdas) max(llambdas)]);
hold on
semilogx(llambdas,sametask(2,:),'-','LineWidth',lwd,'color',lblue);
semilogx(llambdas,hires(2,:)-hires_up(2,:),'-','LineWidth',lwd,'color',dred);
semilogx(llambdas,hires(2,:),'-','LineWidth',lwd,'color',dblue);
hold off
lgd=legend('E[hires]','E[workers at same task]','E[productive workers]');
lgd.Location='southwest';
fig.Units = 'inches';
fig.Position = [0 0 6 2];
set(gca,'FontSize',10);
set(gca,'FontName','CMU Serif');
xlabel('information cost \lambda')
ylabel('# workers')
box off
grid on
exportgraphics(fig,sprintf('%stasks_%1.2fd%1.2fw%1.2f.pdf',d_folder,p,ddelta,wage),'ContentType','vector');
%pause

%% More figures
figure(2)
clf
semilogx(llambdas,[info.MI],'k-','LineWidth',1);
title('Information Acquisition');
%subtitle(sprintf('N = %i, delta = %1.2f, p = %1.2f, wage = %1.2f',N,ddelta,p,wage));
ylim([0,info(1).FI]); xlim([min(llambdas) max(llambdas)]);
hold on
semilogx(llambdas,[info.MI_t],'k--','LineWidth',1);
semilogx(llambdas,[info.MI_w],'k:','LineWidth',1);
hold off
ylabel('mutual information')
lgd=legend('RI','RI + task revelation','RI + worker revelation');
lgd.Location='northeast';
fig=gcf;
fig.PaperSize = sz;
fig.PaperPosition = [0 0 sz];
print(fig,sprintf('%stasksInfo_%1.2fd%1.2fw%1.2f.pdf',d_folder,p,ddelta,wage),'-dpdf');
%pause

figure(3)
clf
semilogx(llambdas,[info.output_E],'k-','LineWidth',1);
title('Output');
%subtitle(sprintf('N = %i, delta = %1.2f, p = %1.2f, wage = %1.2f',N,ddelta,p,wage));
xlim([min(llambdas) max(llambdas)]);
hold on
semilogx(llambdas,[info.output_SD],'k--','LineWidth',1);
semilogx(llambdas,[info.output_H],'k:','LineWidth',1);
semilogx(llambdas,[info.output_SD]./[info.output_E],'b--','LineWidth',1);
hold off
lgd=legend('expectation','volatility','entropy','cv');
lgd.Location='northeast';
fig=gcf;
fig.PaperSize = sz;
fig.PaperPosition = [0 0 sz];
print(fig,sprintf('%stasksOutput_%1.2fd%1.2fw%1.2f.pdf',d_folder,p,ddelta,wage),'-dpdf');

figure(4)
clf
semilogx(llambdas,[info.U],'k-','LineWidth',1);
title('Payoff potential');
%subtitle(sprintf('N = %i, delta = %1.2f, p = %1.2f, wage = %1.2f',N,ddelta,p,wage));
ylim([info(1).U_NI,info(1).U_FI]); xlim([min(llambdas) max(llambdas)]);
hold on
semilogx(llambdas,[info.U_t],'k--','LineWidth',1);
semilogx(llambdas,[info.U_w],'k:','LineWidth',1);
hold off
ylabel('firm profits')
lgd=legend('RI','RI + task revelation','RI + worker revelation');
lgd.Location='southwest';
fig=gcf;
fig.PaperSize = sz;
fig.PaperPosition = [0 0 sz];
print(fig,sprintf('%stasksPayoff_%1.2fd%1.2fw%1.2f.pdf',d_folder,p,ddelta,wage),'-dpdf');
%pause

%% Minority Assignments
d=0.05;
n=4;
tic
fprintf('Group A contains %i workers who are each %2.1f%% more likely to be unproductive than each of the %i workers in group B.\n',n,100*d,N-n);
qn=(1+d)/(N+d*n); q = [qn*ones(n,1); (1-n*qn)/(N-n)*ones(N-n,1)];
[prob,~,~,info] = taskAssignment(8,0.5,q,0.8,0.25,3);
fprintf('Odds of being hired are %1.4f in group A and %1.4f in group B.\n\n',prob.marg*(setup.actions(:,1)>0),prob.marg*(setup.actions(:,N)>0));
toc
