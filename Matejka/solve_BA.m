function [p_marg,hist,ttime,exitflag] = solve_BA(u_mat,ppi,info_cost,varargin)
% solve_BA
%
% Solves Rational Inattention problem using the Blahut-Arimoto algorithm.
%-------------------------------------
% REQUIRED INPUTS:
%  - u_mat: Payoff matrix, with states as rows, actions as columns.
%  - ppi: Prior over states, vector of length(size(u_mat,1)).
%  - info_cost: Strictly positive scalar for information cost lambda.
% See code for additional inputs in varargin.
%
% OUTPUT:
%  - p_marg: Marginal over actions, vector of length(size(u_mat,2))
%  - hist: Computational output, struct object (empty if save_hist = false)
%  - ttime: Running time, in seconds.
%  - exitflag:  numeric, with exit conditions
%     1  Convergence criterion achieved.
%    -1  Number of maximum iterations (maxit) reached.


%% Parsing inputs

inputs = inputParser;

% REQUIRED INPUTS
% Payoff matrix
addRequired(inputs,'u_mat',@(x) isnumeric(x));
[I,J] = size(u_mat); % I: n of states, J: n of actions
action_grid = (1:1:J)';

% Prior over states
isPMF = @(x) isnumeric(x) && (size(x,2)==1) && abs(sum(x)-1)<10^-12 && all(x>=0);
addRequired(inputs,'ppi',@(x) isPMF(x) && size(x,1)==I);

% Information cost
validposn = @(x) isnumeric(x) && isscalar(x) && x>0;
addRequired(inputs,'info_cost',validposn);

% DISPLAY INPUTS
displayoptions = {'off','final','iter','detailed'};
addParameter(inputs,'display','iter',@(x) any(validatestring(x,displayoptions)));
addParameter(inputs,'actionlabels',action_grid,@(x) size(x,1)==J);

% HISTORY
addParameter(inputs,'save_hist',false,@(x) islogical(x));

% STOPPING CRITERION
stopoptions = {'IE','marginal','objective'};
addParameter(inputs,'stopping_rule','objective',...
    @(x) any(validatestring(x,stopoptions)));

% TOLERANCES
addParameter(inputs,'conv_tol_IE',1e-12,validposn);
addParameter(inputs,'conv_tol_marginal',1e-6,validposn);
addParameter(inputs,'conv_tol_obj',1e-12,validposn);
addParameter(inputs,'MaxIterations',1e5, validposn);

% INITIAL GUESS
addParameter(inputs,'initial_p',ones(J,1)/J,@(x) isempty(x) || (isPMF(x) && size(x,1)==J));

% PARSE
parse(inputs,u_mat,ppi,info_cost,varargin{:});

% Some checks
if size(ppi,2)>1
    ppi=ppi'; %make sure ppi is a column vector
end
%check that no two columns are exactly equal (no redundant actions)
if ~prod(size(unique(u_mat','rows'))==size(u_mat'))
    warning("Some actions have identical payoffs across all states.");
end

% Information cost
llambda = info_cost;

% Labels
actionlbls = inputs.Results.actionlabels;

% Stopping criterion
stopping_rule = inputs.Results.stopping_rule;

% Tolerances
conv_tol_IE = inputs.Results.conv_tol_IE;
conv_tol_marginal = inputs.Results.conv_tol_marginal;
conv_tol_obj = inputs.Results.conv_tol_obj;

% maxit limits the number of iterations
maxit = inputs.Results.MaxIterations;

% History
save_hist = inputs.Results.save_hist;

% Print options
print_i = any(strcmp({'iter','detailed'},inputs.Results.display));
print_d = any(strcmp({'detailed'},inputs.Results.display));
print_final = any(strcmp({'final','detailed'},inputs.Results.display));

% Initial guess
marg = inputs.Results.initial_p;
if isempty(marg)
    marg = ones(J,1)/J;
end

%% Setup

chat(print_d,'Computing the transformed B matrix.\n');
% Attention level transformation, including scaling to keep machine
% precision accurate
b_mat = exp(u_mat/llambda);

% Inline functions
IE = @(p) llambda*(log(b_mat*p));
DIE = @(p,q) IE(p) - IE(q);
fun_w = @(p) llambda*ppi'*log(b_mat*p);

% Stopping criterion
if strcmp(stopping_rule,'IE')
    stop_fun = @(p,q) DIE(p,q)'*diag(ppi)*DIE(p,q);
    stop_tol = conv_tol_IE;
elseif strcmp(stopping_rule,'marginal')
    stop_fun = @(p,q) max(abs(p-q));
    stop_tol = conv_tol_marginal;
else
    stop_fun = @(p,q) fun_w(p) - fun_w(q);
    stop_tol = conv_tol_obj;
end

% History
hist = struct;
if save_hist
    hist.margs = marg;
    hist.steps= [];
else
    hist=[];
end

% Initialize
step_size = 1;
exitflag = 1;
ite = 1;

%% Execute

tic
while step_size > stop_tol
    chat(print_i, 'Round %i: ', ite)
    
    % Store old marginal
    marg_old = marg;
    
    % Apply Blahut-Arimoto
    ttemp_mat = repmat(marg',I,1).*b_mat;
    pcond_mat = ttemp_mat./repmat(sum(ttemp_mat,2),1,J);
    marg = pcond_mat'*ppi;
    
    % Compute step size
    step_size = stop_fun(marg,marg_old);
    chat(print_i,'Step size %g.\n',step_size);
    
    % Save history
    if save_hist
        hist.margs(:,end+1) = marg;
        hist.steps(end+1) = step_size;
    end
    
    % Check if maximum iterations have been reached.
    if ite==maxit
        chat(print_i,'Maximum number of iterations reached. Stopping. \n')
        step_size = -1;
        exitflag = -1;
    end
    
    % One more iteration
    ite = ite+1;
end

ttime = toc;

%% Output

% Marginal
p_marg = max(marg,0);
p_marg = p_marg/sum(p_marg);


% Final display
if (print_final)
    fprintf('\n');
    RI_printmarg(p_marg,'actionlabels',actionlbls);
end

end

%% PRINTING FUNCTIONS

% CHAT
function chat(condition,varargin)

if condition
    fprintf(varargin{:});
end

end

function RI_printmarg(marg,varargin)
% RI_PRINTMARG prints a reader-friendly description of chosen actions and marginals to
% the console.
%
% RI_PRINTMARG(p) prints a table with the marginals.
% By default, actions are indexed by the corresponding row in p.
% This is useful if p spans the entire action grid.
% RI_PRINTMARG(p,'actionids',a) supplies a list of action indices for each
% row in p. This is useful if p only contains action within the support.
% RI_PRINTMARG(p,'actionids',a,'actionlabels',alabels) refers to actions by
% their full label (alabels) instead of their index.

% Parse inputs
inputs = inputParser;
addRequired(inputs,'marg',@(x) isnumeric(x) && size(x,2)==1);
J_curr = size(marg,1); %n of chosen actions
addParameter(inputs,'actionids',(1:size(marg,1))',...
    @(x) prod(isnumeric(x)) && (prod(size(x)==[J_curr 1])) && prod(mod(x,1)==0));
addParameter(inputs,'actionlabels',[]);
parse(inputs,marg,varargin{:});

actionids = inputs.Results.actionids;
actionlabels = inputs.Results.actionlabels;

if isempty(actionlabels)
    actionlabels = (1:1:max(actionids))';
end
if (size(actionlabels,1)<max(actionids))
    error("Action index exceeds dimension of the action label list.");
end

% remove empty rows
actionids(marg==0)=[];
marg(marg==0)=[];

content = "";
for k=1:size(marg,1)
    content(k,1) = string(strjoin(string(actionlabels(actionids(k),:))));
end
fieldwidth = max(max(strlength(content)),6)+1;
fprintf('%*s | Probability\n',fieldwidth,"Action");
for k=1:size(marg,1)
    fprintf('%*s | %1.5g \n',fieldwidth,content(k,1),marg(k));
end

end

