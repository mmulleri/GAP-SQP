% RI_Matejka16_probsets.m
%-------------------------------------
% Generates undominated and probability-q sets.
%-------------------------------------

clear;

%% Parameters (Benchmark)

% Constant settings - Model
ufun = @(p,d) p^(-(d+1)/d)*(p-1);
staterange = [1/9 1/2];
actionrange = [10/9 3/2];

% Grids
Nstates = 300;
Nactions = 1000;
stategrid = transpose(staterange(1):(staterange(2)-staterange(1))/(Nstates-1):staterange(2));
actiongrid = transpose(actionrange(1):(actionrange(2)-actionrange(1))/(Nactions-1):actionrange(2));

% Prior
ppi = ones(Nstates,1)/Nstates;

% Information cost
llambda = 0.00531;

% Convergence tolerance for IE
conv_tol_IE = 1e-11;

% Destination folder
d_folder = 'Matejka16_output\';
fig_folder = 'Matejka16_figs\';

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

covers = GAP_bounds(info.b_mat,ppi,p_marg,[0,.01],...
    'verbose',false);

save('Matejka16_output/data_bounds.mat','covers','actiongrid','p_marg');

%% Figure dominated actions
dblue='#143D73';
lblue='#99B1C3';
dorange='#F29F05';
dred='#BF214B';

fig=figure(1); 
clf
fig.Units = 'inches';
fig.Position = [0 0 7 2.5];
hold on 
width = 1;

bar(actiongrid, (1-covers(:,1)), width,...
    'FaceColor',lblue,...
    'EdgeAlpha', 0)
bar(actiongrid, (1-covers(:,2)), width,...
    'FaceColor',dblue,...
    'EdgeAlpha', 0)
bar(actiongrid,p_marg,width,'FaceColor',dorange,'EdgeAlpha', 0)
ylim([0 min(1,1.2*max(p_marg))])

ylabel('probability')
xlabel('price')
set(gca, 'YGrid', 'on', 'XGrid', 'off')

set(gca,'FontSize',9);
set(gca,'FontName','CMU Serif');
main=gca;

% add insets:
% get positions for insets
xl = xlim(); 
yl = ylim(); 
axpos = get(gca,'position');
% convert data coordinate xyc from data space to figure space
% (https://www.mathworks.com/matlabcentral/answers/472527-how-to-align-axes-to-a-point-on-another-figure)
normFigCoord = @(xyc) axpos(1:2) + axpos(3:4).*((xyc- [xl(1),yl(1)])./[xl(2)-xl(1), yl(2)-yl(1)]);

% inset A
inset_yl = [0.1 yl(2)];
inset_lr = 1.185; inset_xstretch=20;los=0.002;
inset_xl_orig = [1.1902 1.1935];
inset_xl = [inset_lr-inset_xstretch*(inset_xl_orig(2)-inset_xl_orig(1)) inset_lr];
inset_ll = normFigCoord([inset_xl(1) inset_yl(1)]);
inset_ur = normFigCoord([inset_xl(2) inset_yl(2)]);
framecoords=[normFigCoord([inset_xl_orig(1) inset_yl(1)]) normFigCoord([inset_xl_orig(2) inset_yl(2)])];
frame = axes('Units','Normalize','Position',[framecoords(1:2) framecoords(3:4)-framecoords(1:2)], 'color', 'none');
set(frame,'XTick',[],'YTick',[]); box on;
inset = axes('Units','Normalize','Position',[inset_ll, inset_ur-inset_ll], 'color', 'none');
hold on
bar(actiongrid, (1-covers(:,1)), width,'FaceColor',lblue,'EdgeAlpha', 0)
bar(actiongrid, (1-covers(:,2)), width,'FaceColor',dblue,'EdgeAlpha', 0)
bar(actiongrid,p_marg,width,'FaceColor',dorange,'EdgeAlpha', 0)
hold off
ylabel(''); xlabel('');
box on
xlim(inset_xl_orig); xticks([1.191 1.193]);
ylim(inset_yl);
set(gca,'ytick',[])
set(gca,'FontSize',9);
set(gca,'FontName','CMU Serif');
set(gca, 'Layer', 'top');
% add guide lines
plot(main,[inset_xl_orig(1)-los inset_lr+los],[inset_yl(1) inset_yl(1)],'k-');
plot(main,[inset_xl_orig(1)-los inset_lr+los],[inset_yl(2) inset_yl(2)],'k-');

% inset B
inset_lr = 1.35;
inset_xl_orig = [1.363 1.366];
inset_xl = [inset_lr-inset_xstretch*(inset_xl_orig(2)-inset_xl_orig(1)) inset_lr];
inset_ll = normFigCoord([inset_xl(1) inset_yl(1)]);
inset_ur = normFigCoord([inset_xl(2) inset_yl(2)]);
framecoords=[normFigCoord([inset_xl_orig(1) inset_yl(1)]) normFigCoord([inset_xl_orig(2) inset_yl(2)])];
frame = axes('Units','Normalize','Position',[framecoords(1:2) framecoords(3:4)-framecoords(1:2)], 'color', 'none');
set(frame,'XTick',[],'YTick',[]); box on;
inset = axes('Units','Normalize','Position',[inset_ll, inset_ur-inset_ll], 'color', 'none');
hold on
bar(actiongrid, (1-covers(:,1)), width,'FaceColor',lblue,'EdgeAlpha', 0)
bar(actiongrid, (1-covers(:,2)), width,'FaceColor',dblue,'EdgeAlpha', 0)
bar(actiongrid,p_marg,width,'FaceColor',dorange,'EdgeAlpha', 0)
hold off
ylabel(''); xlabel('');
box on
xlim(inset_xl_orig); xticks([1.364 1.366]);
ylim(inset_yl);
set(gca,'ytick',[])
set(gca,'FontSize',9);
set(gca,'FontName','CMU Serif');
set(gca, 'Layer', 'top');
% add guide lines
plot(main,[inset_xl_orig(1)-los inset_lr+los],[inset_yl(1) inset_yl(1)],'k-');
plot(main,[inset_xl_orig(1)-los inset_lr+los],[inset_yl(2) inset_yl(2)],'k-');

lgd = legend(main,'99% cover','undominated','numerical solution','','','',...
    'Location','NorthEast');
set(lgd,'FontSize',8);
set(gca, 'Layer', 'top');
figure_file = [fig_folder sprintf('marginals_covers_%i_%i.pdf',Nstates,Nactions)];
exportgraphics(fig,figure_file,'ContentType','vector');
close

%eof