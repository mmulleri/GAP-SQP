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

% Destination folder
d_folder = 'Matejka16_output\';

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

Db_vec = zeros(N_llambda,1); % Distance in vector b
DIE_vec = Db_vec; % Distance in IE

% Initial guess - Full information
[~,FI_actions] = max(u_mat,[],2);
FI_pjoint = zeros(Nstates,Nactions);
FI_pjoint(sub2ind([Nstates Nactions],1:Nstates,FI_actions'))=1;
p_FI = FI_pjoint'*ppi;

% Execute
fprintf('Computing across information costs... \n')
for i = 1:N_llambda
    fprintf(['Step %i of ' num2str(N_llambda) '.\n'],i)
    
    % PREP THIS ITERATION
    llambda = llambda_grid_bench(i);
    % Matrix of b's
    b_mat = exp(u_mat/llambda);
    % Negative of the objective function
    neg_w = @(p) -llambda*ppi'*log(b_mat*p);
    % Ignorance equivalent
    IE = @(p) llambda*log(b_mat*p);
    
    % SQP ALGORITHM
    [p_marg,~,ttime,exitflag] = GAP_SQP(u_mat,ppi,llambda,...
        'display','off',...
        'conv_tol_IE',conv_tol_IE,...
        'save_hist', false);
    b = b_mat*p_marg;
    
    times_mat(i,1) = ttime;
    p_marg = max(p_marg,0);
    p_marg = p_marg/sum(p_marg);
    obj_mat(i,1) = - neg_w(p_marg);
    flag_mat(i,1) = exitflag;
    
    % ALTERNATIVE
    [p_marg,hist,ttime,exitflag] = solve_BA(u_mat,ppi,llambda,...
        'display','off',...
        'stopping_rule','objective',...
        'save_hist', false);
    
    flag_mat(i,2) = exitflag;
    p_marg = max(p_marg,0);
    p_marg = p_marg/sum(p_marg);
    b_X = b_mat*p_marg;
    times_mat(i,2) = ttime;
    obj_mat(i,2) = -neg_w(p_marg);
    
    % METRIC DIFFERENCES
    Db_vec(i) = (b-b_X)'*diag(ppi)*(b-b_X);
    dIE = IE(p_marg)-IE(p_marg);
    DIE_vec(i) = dIE'*diag(ppi)*dIE;
end
fprintf('Completed. \n')

% Save output
save([d_folder 'data_runtimes_infocosts.mat'])

%% Increase grid precision in step for actions and states

times_mat = zeros(Ngrid,2);% Running times
obj_mat = times_mat; % Objective function
flag_mat = times_mat; % Exitflags

Db_vec = zeros(Ngrid,1); % Distance in vector b
DIE_vec = Db_vec; % Distance in IE

fprintf('Computing across grid precision values... \n')
for n=1:Ngrid
    fprintf(['Step %i of ' num2str(Ngrid) '.\n'],n)
    
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
    
    
    % SQP ALGORITHM
    [p_marg,~,ttime,exitflag] = GAP_SQP(u_mat,ppi,llambda,...
        'display','off',...
        'conv_tol_IE',conv_tol_IE,...
        'save_hist', false);
    b = b_mat*p_marg;
    times_mat(n,1) = ttime;
    p_marg = max(p_marg,0);
    p_marg = p_marg/sum(p_marg);
    obj_mat(n,1) = - neg_w(p_marg);
    flag_mat(n,1) = exitflag;
    
    % BA ALGORITHM
    [p_marg,~,ttime,exitflag] = solve_BA(u_mat,ppi,llambda,...
        'display','off',...
        'stopping_rule','objective',...
        'save_hist', false);
    flag_mat(n,2) = exitflag;
    p_marg = max(p_marg,0);
    p_marg = p_marg/sum(p_marg);
    b_X = b_mat*p_marg;
    times_mat(n,2) = ttime;
    obj_mat(n,2) = -neg_w(p_marg);
    flag_mat(n,2) = exitflag;
    
    % Differences
    Db_vec(n) = (b-b_X)'*diag(ppi)*(b-b_X);
    dIE = IE(p_marg)-IE(p_marg);
    DIE_vec(n) = dIE'*diag(ppi)*dIE;
end
fprintf('Completed. \n')

% Save output
save([d_folder 'data_runtimes_gridpoints.mat'])
