function [u_mat,ppi] = GAP_discretize(ufun,actiongrid,stategrid,statedist)
  % RI_DISCRETIZE(ufun,actiongrid,stategrid,statedist)
  % discretizes continuous choice problems.
  %
  % It requires the following inputs:
  % - ufun is an anonymous function with two arguments. To define, write
  %     ufun = @(x,s) *utility_of_action_x_in_state_s*;
  %   by replacing the starred placeholder with the desired utility function,
  %   e.g. ufun = @(x,s) log(x-s);
  % - actiongrid is a column vector that describes the desired discretization
  %   of the action space.
  % - stategrid is a column vector that describes the desired discretization
  %   of the state space.
  % - statedist describes the prior distribution over states.
  %   Two formats are accepted:
  %   - if statedist is a nonnegative numerical vector of the same length as
  %     stategrid, it is interpreted as a discrete probability mass function.
  %   - if statedist is an anonymous function with a single argument, it is
  %     interpreted as a continuous CDF. The discrete approximation will
  %     attribute the likelihood of any state to the closest gridpoint on
  %     stategrid.
  % The function returns a payoff matrix u_mat and a discretized prior
  % distribution (equal to statedist for discrete state distributions).
  % In turn, these can be fed into the companion function RI_solve.
  %
  %
  % written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.
  % For further details, see Armenter-et-al, 2019.
  %
  inputs = inputParser;
  addRequired(inputs,'ufun',@(x) isa(x,'function_handle') && nargin(x)==2);
  addRequired(inputs,'actiongrid',@(x) isnumeric(x));
  J = length(actiongrid); %n of actions
  addRequired(inputs,'stategrid',@(x) isnumeric(x));
  I = length(stategrid); %n of states
  isPMF = @(x) isnumeric(x) && all(sort(size(x))==[1 I]) && abs(sum(x)-1)<10^-12 && all(x>=0);
  isCDF = @(x) isa(x,'function_handle') && nargin(x)==1;
  addRequired(inputs,'statedist',@(x) isPMF(x) || isCDF(x));
  parse(inputs,ufun,actiongrid,stategrid,statedist);
  if isPMF(statedist)
    ppi=statedist;
  else %statedist is a CDF
    ppi=[statedist(mean(stategrid([1 2])))];
    for i=2:I-1
      ppi(end+1) = statedist(mean(stategrid([i i+1])))-statedist(mean(stategrid([i i-1])));
    end
    ppi(I)= 1-statedist(mean(stategrid([I-1 I])));
  end
  if size(ppi,2)>1
    ppi=ppi'; %make sure ppi is a column vector
  end

  u_mat = zeros(I,J);
  for s=1:I
    for x=1:J
      u_mat(s,x) = ufun(actiongrid(x,:),stategrid(s,:));
    end
  end
end
