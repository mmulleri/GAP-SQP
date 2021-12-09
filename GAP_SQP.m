function [p_marg,hist,ttime,exitflag,info] = GAP_SQP(u_mat,ppi,info_cost,varargin)
  % RI_QUAD
  % This code uses quadratic approximation to solve a rational inattention problem
  % based on the geometric target problem from Armenter et al, 2019.
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
  %     1  Change in IE less than conv_tol_IE.
  %     0  Quadratic approximation cannot improve objective function.
  %    -1  Number of maximum iterations (maxit) reached.
  %
  % For further details, see Armenter-et-al, 2019.
  %
  % written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.


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

  % Other input parameters
  addParameter(inputs,'actionlabels',action_grid,@(x) size(x,1)==J);
  addParameter(inputs,'save_hist',false,@(x) islogical(x));
  displayoptions = {'off','none','final','iter','detailed'};
  addParameter(inputs,'display','iter',@(x) any(validatestring(x,displayoptions)));

  % Computational
  addParameter(inputs,'conv_tol_IE',1e-12,validposn);
  addParameter(inputs,'MaxIterations',1e4,validposn);
  addParameter(inputs,'initial_p',[],@(x) isempty(x) || (isPMF(x) && size(x,1)==J));
  addParameter(inputs,'zero_tol',1e-9,@(x) isnumeric(x) && isscalar(x) && x>=0); % algorithm will disregard actions with joint probability below threshold zero_tol

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

  % save_hist specifies whether or not to save evaluated marginals.
  save_hist = inputs.Results.save_hist;

  % conv_tol_IE is the convergence tolerance in the IE
  conv_tol_IE = inputs.Results.conv_tol_IE;

  % maxit limits the number of iterations FOR OUR ALGORITHM
  maxit = inputs.Results.MaxIterations;

  % Print options
  print_i = any(strcmp({'iter','detailed'},inputs.Results.display));
  print_d = any(strcmp({'detailed'},inputs.Results.display));
  print_final = any(strcmp({'final','detailed'},inputs.Results.display));

  % Initial guess
  marg = inputs.Results.initial_p;

  % zero tolerance
  zerotol = inputs.Results.zero_tol;
  %% Computational parameters

  linoptions = optimoptions('linprog',...
  'Algorithm','dual-simplex',...
  'Display','off',...
  'MaxIterations',10000,...
  'MaxTime',120,...
  'OptimalityTolerance',1e-10,...
  'ConstraintTolerance',1e-9);

  quadoptions = optimoptions('quadprog',...
  'Display','none',...
  'MaxIterations',200,...
  'ConstraintTolerance',1e-9,...
  'OptimalityTolerance', 1e-10);

  %% Setup

  chat(print_d,'Computing the transformed B matrix.\n');
  % Attention level transformation, including scaling to keep machine
  % precision accurate
  b_logscales = -max(u_mat,[],2)/llambda;
  b_mat = exp(u_mat/llambda + b_logscales);

  % STARTING GUESS
  % Now, we choose an initial point. We find that starting with
  % full-information actions typically works well, though the researcher has
  % some flexibility here
  if isempty(marg)
    chat(print_d,'Computing the full-information solution.\n');
    [~,FI_actions] = max(u_mat,[],2);
    FI_pjoint = zeros(I,J); FI_pjoint(sub2ind([I J],1:I,FI_actions'))=1;
    marg = FI_pjoint'*ppi;
  end
  b = b_mat*marg;

  % HISTORY
  hist = struct;
  if save_hist
    hist.bs = [];
    hist.slide = [];
    hist.margs = marg;
    hist.flags = [];
  else
    hist=[];
  end

  % NEGATIVE OF THE OBJECTIVE FUNCTION
  neg_w = @(p) -llambda*ppi'*log(b_mat*p);

  % IGNORANCE EQUIVALENT
  IE = @(p) llambda*(log(b_mat*p)-b_logscales);
  ppiD = spdiags(ppi,0,length(ppi),length(ppi));
  IEdist = @(dIEvec) dIEvec'*ppiD*dIEvec;

  % INITIALIZE
  i=1;
  stepsize=1;
  exitflag = 1;
  tic

  %% Execute
  while stepsize > conv_tol_IE
    b_old=b;

    chat(print_i, 'Round %3i: ',i);
    % compute z scores & disregard unlikely actions:
    scores=(ppi./b)'*b_mat-1;
    zmax=max(scores);
    cand=(scores>= min(-(1-zerotol)/zerotol*zmax,-zerotol)); %added alternative threshold to include more actions, primarily for cases where numerical imprecision causes zmax<0.
    j=sum(cand);



    % use second-order taylor approximation
    D = spdiags(ppi./(b.*b),0,length(ppi),length(ppi));
    H = b_mat(:,cand)'*D*b_mat(:,cand);

    marg_old=marg;
    marg_trimmed=marg(cand);

    if (zerotol>0)
        chat(print_d, 'Old marginals associate weight %e to %i actions now disregarded. ',sum(marg_old(~cand)),J-j);
        chat(print_d, 'Max score is %e and effective cutoff is %e.\n',zmax,min(-(1-zerotol)/zerotol*zmax,-zerotol));
        chat(print_d, 'Calling quadprog with a %i-dimensional problem.\n',j);
    end

    [Dmarg,~,found_quad,~] = quadprog((H+H')/2, ...
        (-2*(ppi./b)'*b_mat(:,cand) + marg_trimmed'*(H+H')/2)',...
        [],[],ones(1,j),0, zeros(j,1) - marg_trimmed,ones(j,1)-marg_trimmed, ...
        [],quadoptions);
    chat(print_d,'Quadprog terminated with exitflag %i. \n',found_quad)
    if size(Dmarg,2)<1
      chat(print_i,'Quadratic approximation cannot improve objective function. Stopping. \n')
      marg = marg_old;
      exitflag = 0;
      stepsize = -1;
      continue
    end

    marg(cand) = marg(cand) + Dmarg;
    Dw = -neg_w(marg) + neg_w(marg_old);
    slide_step = false;
    % Report diagnostics
    if Dw>=0
      chat(print_d,'Quadprog has improved objective by %e. Continuing. \n',Dw)
      slide_step = true;
    end
    found_quad = max(found_quad,0);

    % Slide
    if slide_step
      b_old = b;
      b_new = b_mat*marg;
      [b,t]=slide(b_old,b_new);
    end

    if save_hist
      hist.margs(:,end+1) = marg;
      hist.slide(:,end+1) = t;
      hist.bs(:,end+1) = b;
      hist.flags(:,end+1) = [found_quad;found_gi];
    end

    if exitflag == 0
      chat(print_i,'Quadratic approximation cannot improve objective function. Stopping. \n')
      stepsize = -1;
    else
      stepsize = IEdist(IE(marg)-IE(marg_old));
      chat(print_i,'Iteration has step size %g.\n',stepsize);
      if i==maxit
        stepsize = -1;
        chat(print_d,'Maximum number of iterations reached. Stopping. \n')
        exitflag = -1;
      end
      i=i+1;
    end

    if (print_d)
        fprintf('\n');
        GAP_printmarg(marg,'actionlabels',actionlbls);
    end

  end

  % Scaling now
  constr1 = [-b_mat,b];
  ff = [zeros(J,1);-1];
  chat(print_d,'Scaling final estimate to the surface of B. \n');
  [th,~,found_scale] = linprog(ff,constr1,...
  zeros(I,1),[ones(1,J),0],1,[zeros(J,1);0],[ones(J,1);Inf],...
  linoptions);

  if found_scale==1
    chat(print_d,'Scaling factor is 1+%6g. \n',th(end)-1);
    p_marg = th(1:(end-1));
  else
    chat(print_d,'Scaling unsuccessful, maintaining quadprog marginals.');
    p_marg = marg; %use last one
  end

  ttime = toc;

  %% FINAL OUTPUT
  % make sure marginals are a valid probability
  p_marg = max(0,p_marg);
  p_marg = p_marg/sum(p_marg);

  if (print_final)
    fprintf('\n');
    GAP_printmarg(p_marg,'actionlabels',actionlbls);
  end

  if nargout>4
    info.b_mat = b_mat;
    info.b_logscales = b_logscales;
  end

  %% FUNCTIONS

  % SLIDE
  function [b,t] = slide(b_start,b_end)
    if (ppi./b_end)'*(b_start-b_end)<=0
      b = b_end;
      t = 1;
    elseif (ppi./b_start)'*(b_start-b_end)>=0
      b = b_start;
      t = 0;
    else
      t=fzero(@(t) (ppi./(t*b_end+(1-t)*b_start))'*(b_start-b_end), [0 1]);
      % find point where upper contour set is tangent to segment [b_start,b_end]
      b=t*b_end+(1-t)*b_start;
    end
  end


  % CHAT
  function chat(condition,varargin)
    if condition
      fprintf(varargin{:});
    end
  end

end
