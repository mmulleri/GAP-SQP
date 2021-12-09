% RI_Matejka16_main.m
%-------------------------------------
% Document running times for GAP-SQP and BA for Section 4.1.
%-------------------------------------
% NOTE: Requires to run RI_Matejka16_init.m first.

clear;

%% Model parameters

% Constant settings - Model
ufun = @(p,d) p^(-(d+1)/d)*(p-1);
staterange = [1/9 1/2];
actionrange = [10/9 3/2];

% Information cost
llambda = 0.00531;


%% Computation parameters

% Convergence tolerance for IE
conv_tol_IE = 1e-11;

% Number of runs to be averaged over
K=10; 
    
% Destination folder
d_folder = 'Matejka16_output/';

% Benchmark grid points
Nstates_bench = 200;
Nactions_bench = 200;

% Grid for information costs (benchmark)
N_llambda = 50;
llambda_grid_bench = linspace(.00025,.01, N_llambda);

% Grid points considered
gridpoints = 50:25:600;
Ngrid = length(gridpoints);

%% Run times across information cost values

Nstates = Nstates_bench;
Nactions = Nactions_bench;

% Setup
stategrid = transpose(staterange(1):(staterange(2)-staterange(1))/(Nstates-1):staterange(2));
ppi = ones(Nstates,1)/Nstates;
actiongrid = transpose(actionrange(1):(actionrange(2)-actionrange(1))/(Nactions-1):actionrange(2));

u_mat = zeros(Nstates,Nactions);
for s=1:Nstates
    for x=1:Nactions
        u_mat(s,x) = ufun(actiongrid(x),stategrid(s));
    end
end

% Initialize
times_mat = zeros(N_llambda,2); % Running times
obj_mat = times_mat; % Objective function
flag_mat = times_mat; % Exitflags

% Initial guess - Full information
[~,FI_actions] = max(u_mat,[],2);
FI_pjoint = zeros(Nstates,Nactions);
FI_pjoint(sub2ind([Nstates Nactions],1:Nstates,FI_actions'))=1;
p_FI = FI_pjoint'*ppi;

% Execute
fprintf('Computing across information costs... \n')
PI=randperm(N_llambda);
for j = 1:N_llambda
    i=PI(j);

    % PREP THIS ITERATION
    llambda = llambda_grid_bench(i);
    fprintf(['Step %i of ' num2str(N_llambda) ', lambda = %g.\n'],j,llambda);
    % Matrix of b's
    b_mat = exp(u_mat/llambda);
    % Negative of the objective function
    neg_w = @(p) -llambda*ppi'*log(b_mat*p);
    % Ignorance equivalent
    IE = @(p) llambda*log(b_mat*p);
    
    [times_mat(i,:),flag_mat(i,:),obj_mat(i,:)] = RI_timed_runs(u_mat,b_mat,ppi,llambda,conv_tol_IE,neg_w);
end
fprintf('Completed. \n')

% Save output
save([d_folder 'data_runtimes_infocosts.mat'])

%% Increase grid precision in step for actions and states

times_mat = zeros(Ngrid,2);% Running times
obj_mat = times_mat; % Objective function
flag_mat = times_mat; % Exitflags

fprintf('Computing across grid precision values... \n')
PI=randperm(Ngrid);
for m=1:Ngrid
    n=PI(m);
    fprintf(['Step %i of ' num2str(Ngrid) ', grid size %i.\n'],m,n);

    % Assign precision parameters
    Nstates = gridpoints(n);
    Nactions = gridpoints(n);

    % State grid
    stategrid = transpose(staterange(1):(staterange(2)-staterange(1))/(Nstates-1):staterange(2));
    % Prior (Uniform)
    ppi = ones(Nstates,1)/Nstates;
    % Action grid
    actiongrid = transpose(actionrange(1):(actionrange(2)-actionrange(1))/(Nactions-1):actionrange(2));
    % Payoff matrix
    u_mat = zeros(Nstates,Nactions);
    for s=1:Nstates
        for x=1:Nactions
            u_mat(s,x) = ufun(actiongrid(x),stategrid(s));
        end
    end
    % Matrix of b's
    b_mat = exp(u_mat/llambda);
    % Negative of the objective function
    neg_w = @(p) -llambda*ppi'*log(b_mat*p);
    % Ignorance equivalent
    IE = @(p) llambda*log(b_mat*p);

    [times_mat(n,:),flag_mat(n,:),obj_mat(n,:)] = RI_timed_runs(u_mat,b_mat,ppi,llambda,conv_tol_IE,neg_w);
end
fprintf('Completed. \n')

% Save output
save([d_folder 'data_runtimes_gridpoints.mat'])
