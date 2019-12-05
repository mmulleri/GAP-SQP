function [probsets] = GAP_bounds(b_mat,ppi,popt,plevels,varargin)
  % RI_BOUNDS returns bounds on the action space
  % As inputs, the function expects:
  % - b_mat: a matrix of transformed payoffs (provided by GAP_SQP)
  %     b_mat can be equal to [] as long as actions, u and b_logscales are supplied
  % - ppi: a vector of marginals over the state space
  % - popt: a candidate optimum given as a column vector (provided by GAP_SQP)
  % - plevels: a vector of desired probability levels:
  %   p=-1 returns actions that are interior to the learning-proof menu
  %      These actions are never chosen from the current menu, even if
  %      additional actions are added or the prior distribution changes.
  %   p=0 returns dominated actions that are never chosen in the current menu
  %   p>0 returns actions that jointly are chosen with less than p probability
  % - optional arguments:
  %   - seed: distance around which additional points are sampled in order to
  %     bound the location of the IE as tightly as possible
  %   - ps: a matrix where each column represents marginals over the action
  %     space (e.g. evalps provided by GAP_SQP)
  %   - zerotolerance: a small nonnegative scalar that lowers the threshold
  %     for inclusion into the 1-cover
  % As output, the function returns a binary indicator matrix where each
  % column corresponds to a set of actions with the desired probability bounds.
  %
  %
  % written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.
  % For further details, see Armenter-et-al, 2019.
  %
  inputs = inputParser;
  addRequired(inputs,'b_mat',@(x) isnumeric(x) && all(all(x>=0)));
  I = size(b_mat,1); %n of states
  J = size(b_mat,2); %n of actions
  validpmfs = @(x) isnumeric(x) && all(abs(sum(x,1)-1)<1e-12) && all(all(x>=0));
  addRequired(inputs,'ppi',@(x) validpmfs(x) && all(size(x)==[I 1],2));
  addRequired(inputs,'popt',@(x) validpmfs(x) && all(size(x)==[J 1],2));
  addRequired(inputs,'plevels',@(x) isnumeric(x) && min(size(plevels))==1 && all((x>=0 & x<=1)|x==-1));
  addParameter(inputs,'ps',[],@(x) validpmfs(x) && size(x,1)==J);
  displayoptions = {'off','detailed'};
  addParameter(inputs,'verbose',false,@(x) islogical(x));
  addParameter(inputs,'seed',1e-3,@(x) isnumeric(x) && isscalar(x) && (x>=0));
  addParameter(inputs,'zerotolerance',1e-12,@(x) isnumeric(x) && isscalar(x) && (x>=0));
  parse(inputs,b_mat,ppi,popt,plevels,varargin{:});
  if size(plevels,1)>1 %make plevels into a row vector
    plevels=plevels';
  end
  ps = inputs.Results.ps;
  if length(ps)==0
    ps=popt;
  end
  verbose = inputs.Results.verbose;
  numzero = inputs.Results.zerotolerance;
  db=inputs.Results.seed;

  if max(plevels)<0 %no perturbations are needed
    db=0;
  end

  if db>0 % add additional marginals by slightly varying bopt in all dimensions
    if verbose
      fprintf('Generating perturbations from the provided estimate.\n');
    end
    b0=b_mat*popt;
    ps=[ps,ones(J,I)];
    parfor i=1:I
      btilde=b0;
      btilde(i)=b0(i)+db;
      constr1 = [-b_mat,btilde];
      ff = [zeros(J,1);-1];
      % scale btilde to the surface:
      [paug_opt,neg_ray_scale,exitflag] = linprog(ff,constr1,zeros(I,1),[ones(1,J),0],1,[zeros(J,1);0],[ones(J,1);Inf],optimset('Display','off'));
      if (exitflag==1)
        ray_scale = -neg_ray_scale;
        ps(:,i) = paug_opt(1:end-1);
      end
    end
  end
  npoints=size(ps,2);
  ps=max(ps,0)./sum(ps,1); % make sure they are valid probabilities

  probsets=zeros(J,size(plevels,2));
  for k=1:size(plevels,2)
    plevel=plevels(k);
    if (plevel<=0) % determine dominated actions
      if verbose
        fprintf('Looking for dominated actions.\n');
      end
      psi=optimvar('psi',I,'LowerBound',0);
      prob_psi = optimproblem();
      prob_psi.Constraints.ineq = (b_mat'*psi <= ones(J,1));

      if (plevel==0) % impose feasibility bounds from estimates
        bs = b_mat * ps;
        psis = ppi./bs;

        p=optimvar('p',J,'LowerBound',0,'UpperBound',1);
        prob_p = optimproblem();
        prob_p.Constraints.ineq = (psis'*b_mat*p >= ones(npoints,1));
        prob_p.Constraints.sum = (ones(1,J)*p == 1);

        %add dual bounds obtained by minimal feasible b_i in each dimension
        bpsi_UB=Inf(I,1);
        bpsi_LB=zeros(I,1);
        parfor i=1:I
          prob_pi=prob_p;
          %find maximal and minimal feasible b_i
          prob_pi.Objective= b_mat(i,:)*p;
          minbi = prob2struct(prob_pi);
          minbi.options=optimset('Display',"none");
          [pmin,bimin,found]=linprog(minbi);
          if found==1
            bpsi_UB(i) = ppi(i)/bimin;
          else
            if verbose
              warning('Could not determine an upper bound in dimension %i. Exitflag %i.\n',i,found);
            end
          end
          prob_pi.Objective= -b_mat(i,:)*p;
          maxbi = prob2struct(prob_pi);
          maxbi.options=optimset('Display',"none");
          [pmax,bimax,found]=linprog(maxbi);
          if found==1
            bpsi_LB(i) = -ppi(i)/bimax;
          else
            if verbose
              warning('Could not determine a lower bound in dimension %i. Exitflag %i.\n',i,found);
            end
          end
        end
        psi.UpperBound = bpsi_UB;
        psi.LowerBound = bpsi_LB;
      end
      % compute upper bound for feasible z scores
      Z0 = zeros(J,1);
      parfor j=1:J
        prob_psi_j=prob_psi;
        prob_psi_j.Objective= -b_mat(:,j)'*psi;
        maxzj = prob2struct(prob_psi_j);
        maxzj.options=optimset('Display',"none");
        [psizj,zj,found] = linprog(maxzj);
        if found==1
          Z0(j) = -zj-1;
          if verbose
            fprintf('Action %i has a maximum score of %5g.\n',j,-zj-1);
          end
        else
          warning('Could not determine maximum for action %i. Exitflag %i.\n',j,found);
        end
      end
      probsets(:,k) = (Z0<-numzero); % eliminate those with negative z-score
    else %plevel is positive: identify all actions with small enough z scores
      if verbose
        fprintf('Looking for unlikely actions.\n');
      end
      psi0=ppi./(b_mat*popt);
      z=psi0'*b_mat-1;
      Delta = max(z)
      delta = Delta * (1-plevel)/plevel
      probsets(:,k) = (z<-delta);
    end
  end
end
