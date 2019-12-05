function [p_cond,p_joint,uopt,Iopt,u_info] = GAP_components(p_marg,u_mat,llambda,ppi)
  % GAP_COMPONENTS(p_marg,u_mat,llambda,ppi) describes the optimal choice
  % under marginals p_marg.
  %
  %-------------------------------------
  % REQUIRED INPUTS: (all provided by GAP_SQP)
  %  - p_marg: optimal marginals
  %  - u_mat: payoff matrix with states as rows, actions as columns.
  %  - llambda: Strictly positive scalar for information cost lambda.
  %  - ppi: Prior over states, vector of length(size(u_mat,1)).
  %
  % OUTPUT:
  %  - p_cond: matrix where entry (i,a) denotes the conditional probability of
  %            action a in state i
  %  - p_joint: matrix where entry (i,a) denotes the joint probability of
  %             action a and state i
  %  - uopt:    optimal expected consumption utility
  %  - Iopt:    optimal amount of mutual information (in nats)
  %  - u_info:  structure with three fields
  %     - full_info: consumption payoff under free full Information
  %     - no_info:   consumption payoff under no Information
  %     - normalized: normalized net utility under costly information acquisition
  %       0 means the agent is no better off than under no information
  %       1 means the agent is no worse off than under free full information
  %
  % For further details, see Armenter-et-al, 2019.
  %
  % written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.
  %
  
[I J]=size(b_mat);
b_logscales = -max(u_mat,[],2)/llambda;
b_mat = exp(u_mat/llambda + b_logscales);

p_marg=max(p_marg,0);
p_marg=p_marg/sum(p_marg);
p_cond=zeros(I,J);
p_marg_nzero = p_marg(p_marg>0);
D = spdiags(p_marg_nzero,0,length(p_marg_nzero),length(p_marg_nzero));
p_cond_nzero = b_mat(:,p_marg>0)*D;
p_cond_nzero = p_cond_nzero./sum(p_cond_nzero,2); % Conditional distribution
p_cond(:,p_marg>0)=p_cond_nzero;
D = spdiags(ppi,0,length(ppi),length(ppi));
p_joint = D*p_cond; % Joint distribution
q = @(pjoint) pjoint./(sum(pjoint,1).*sum(pjoint,2));
I = @(p,q) sum(p(p>0).*log(q(p>0)));
Iopt = I(p_joint,q(p_joint));
uopt = sum(sum(p_joint.*u_mat))-llambda*Iopt;
if nargout>4
    u_info = struct;
    u_info.full_info = ppi'*max(u_mat,[],2);
    u_info.no_info = max(ppi'*u_mat);
    u_info.normalized = (uopt-u_info.no_info)/(u_info.full_info-u_info.no_info);
end
end
