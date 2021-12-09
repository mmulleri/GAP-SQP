function [prob,ttime,exitflag,info,setup] = taskAssignment(N,p,q,ddelta,wage,info_cost,varargin)
% Set up and solve an instance of the task assignment problem. For
% details, see Armenter-et-al, 2021.
%-------------------------------------
% REQUIRED INPUTS:
%  - N: number of workers
%  - p: prior likelihood that task 1 is beneficial
%  - q: column vector of prior likelihood that each worker is unproductive
%  - ddelta: marginal contribution of first worker
%  - wage: fixed cost per hired worker
%  - info_cost: Strictly positive scalar for information cost lambda.
%
% See code for additional inputs in varargin.
%
% OUTPUT:
%  - prob.marg: Marginal choice probabilities (row vector)
%  - prob.joint: Joint probabilities (matrix)
%  - prob.post: Posterior beliefs at each action (matrix)
%  - ttime: Running time, in seconds.
%  - exitflag:  numeric, with exit conditions
%     1  Change in IE less than conv_tol_IE.
%     0  Quadratic approximation cannot improve objective function.
%    -1  Number of maximum iterations (maxit) reached.
%  - info: further outputs, see code.
%
% For further details, see Armenter-et-al, 2019.
%
% written by Roc Armenter, Michele Muller-Itten and Zachary Stangebye.

%% Parsing inputs

inputs = inputParser;

% REQUIRED INPUTS
addRequired(inputs,'N',@(x) isnumeric(x) && 0<x && rem(x,1)==0);
addRequired(inputs,'p',@(x) isscalar(x) && 0<=x && x<=1);
addRequired(inputs,'q',@(x) isnumeric(x) && size(q,2)==1 && all(x>=0));
addRequired(inputs,'ddelta',@(x) isscalar(x) && 0<=x && x<=1);
addRequired(inputs,'wage',@(x) isscalar(x) && 0<=x);
addRequired(inputs,'info_cost',@(x) isscalar(x) && 0<x);

% Other input parameters
isPMF = @(x) isnumeric(x) && (min(size(x))==1) && abs(sum(x)-1)<10^-12 && all(x>=0);
addParameter(inputs,'setup',[]); %to reuse previously computed setup parameters
addParameter(inputs,'initial_guess',[],@(x) isempty(x) || isPMF(x));

% PARSE
parse(inputs,N,p,q,ddelta,wage,info_cost,varargin{:});
setup = inputs.Results.setup;
marg = inputs.Results.initial_guess;
q = q/sum(q); %normalize q (irrelevant if q is a proper PMF)

%SETUP
assert(size(q,1)==N);
if (~isempty(setup)) %checks if certain elements are pre-supplied
    u_mat=setup.u_mat;
    [I J] = size(u_mat);
    Phi = setup.Phi;
    phi_mat = setup.phi_mat;
    ppi = setup.ppi;
    states = setup.states;
    actions = setup.actions;
    actionlbls = setup.actionlabels;
    employed=setup.employed;
    majority=setup.majority;
    productive=setup.productive;
    beneficial=setup.beneficial;
else
    fprintf('Setting up task assignment problem with %i workers. ',N);
    [X,Y] = meshgrid([1 2],1:1:N);
    states = [X(:) Y(:)];
    ppi = [p*q; (1-p)*q];
    actions = dec2base(0:1:3^N-1,3)-'0';
    I = size(states,1);
    J = size(actions,1);
    fprintf('There are %i states and %i actions.\n',I,J);

    % Payoff matrix (states in rows, actions in columns)
    Phi = @(m) (ddelta-ddelta^(m+1))/(1-ddelta);%sum(cumprod(ddelta*ones(1,max(0,m))));
    employed = NaN*zeros(1,J);
    majority = NaN*zeros(1,J);
    productive = NaN*zeros(I,J); %number of productive workers at beneficial task
    beneficial = NaN*zeros(I,J); %number of workers at beneficial task
    for (j=1:J) %action
        employed(j) = sum(actions(j,:)~=0);
        majority(j) = max(sum(actions(j,:)==1),sum(actions(j,:)==2));
        for (i=1:I) %state
            critical = states(i,1);
            skilled = (1:N ~= states(i,2));
            productive(i,j) = sum(actions(j,skilled)==critical);
            beneficial(i,j) = sum(actions(j,:)==critical);
        end
    end
    phi_mat = arrayfun(Phi,productive);
    u_mat = phi_mat-wage*employed;

    for j=1:J %generate easily readable actionlabels
        ids=1:N;
        attask1=ids(actions(j,:)==1);
        attask2=ids(actions(j,:)==2);
        lbl="";
        if (length(attask1)==0)
            lbl = strcat("Task 1: nobody.");
        elseif (length(attask1)==1)
            lbl = strcat("Task 1: Worker ",sprintf('%i.',attask1(end)));
        elseif (length(attask1)==N)
            lbl = strcat("Task 1: Everyone.");
        elseif (length(attask1)==N-1)
            lbl = strcat("Task 1: All but ",sprintf('%i.',setdiff(ids,attask1)));
        else
            lbl = strcat("Task 1: Workers ",sprintf('%i, ',attask1(1:(end-1))),sprintf(' %i.',attask1(end)));
        end
        if (length(attask2)==0)
            lbl = strcat(lbl," Task 2: Nobody.");
        elseif (length(attask2)==1)
            lbl = strcat(lbl," Task 2: Worker ",sprintf('%i.',attask2(end)));
        elseif (length(attask2)==N)
            lbl = strcat(lbl," Task 2: Everyone.");
        elseif (length(attask2)==N-1)
            lbl = strcat(lbl," Task 2: All but ",sprintf('%i.',setdiff(ids,attask2)));
        else
            lbl = strcat(lbl," Task 2: Workers ",sprintf('%i, ',attask2(1:(end-1))),sprintf(' %i.',attask2(end)));
        end
        actionlbls(j,1)=lbl;
    end
end

if (~isempty(marg))
    assert(all(size(marg)==[1 J]));
end

%% Solve RI problem
%shortcut for symmetric
if (p==0.5 & max(q)==min(q))
    tic
    b_logscales = -max(u_mat,[],2)/info_cost;
    b_mat = exp(u_mat/info_cost + b_logscales);
    bxone=ones(1,I)*b_mat;
    prob.marg=zeros(J,1);
    prob.marg(bxone==max(bxone))=1/sum(bxone==max(bxone));
    GAP_printmarg(prob.marg,'actionlabels',actionlbls);
    ttime=toc;
    exitflag=0;
else
    [prob.marg,~,ttime,exitflag]= GAP_SQP(u_mat,ppi,info_cost,'actionlabels',actionlbls, ...
        'display',"final",'conv_tol_IE',1e-9,'initial_p',marg);
end
[~,prob.joint,~,Iopt,~] = GAP_components(prob.marg,u_mat,info_cost,ppi);
prob.joint=prob.joint/sum(prob.joint,'all');
prob.marg=sum(prob.joint,1); %avoids numerical inconsistency
prob.post = prob.joint./(ones(I,1)*sum(prob.joint,1));

% save description of optimal choice in info struct
info.nats = Iopt;
info.hired = [0:N; arrayfun(@(n) sum(prob.marg(employed==n)),0:N)];
info.productive = [0:N-1; arrayfun(@(n) sum(prob.joint(productive==n)),0:N-1)];
info.Nhired = [min(employed(prob.marg>1e-9)) ; employed * prob.marg'; max(employed(prob.marg>1e-9))]; %min E max hires
info.Nsametask = [min(majority(prob.marg>1e-9)) ; majority * prob.marg'; max(majority(prob.marg>1e-9))]; %min E max workers at the same task
info.Nprod = [min(min(productive(:,prob.marg>1e-9))) ; sum(prob.joint.*productive,'all') ; max(max(productive(:,prob.marg>1e-9)))];  %min E max number of productive workers
unproductive=ones(I,1)*employed-productive;
info.Nunprod = [min(min(unproductive(:,prob.marg>1e-9))) ; sum(prob.joint.*unproductive,'all') ; max(max(unproductive(:,prob.marg>1e-9)))];  %min E max number of unproductive workers
info.Nbtask = [min(min(beneficial(:,prob.marg>1e-9))) ; sum(prob.joint.*beneficial,'all') ; max(max(beneficial(:,prob.marg>1e-9)))];  %min E max number of unproductive workers

%% compute output improvements with ex-post information shock about tasks/workers:
post_eff=prob.post(:,prob.marg>1e-9);   % realized posteriors
dpost_eff=prob.marg(prob.marg>1e-9); % marginal likelihood of each posterior
dpost_eff=dpost_eff/sum(dpost_eff); %renormalize
info.U= max(u_mat'*post_eff)*dpost_eff';
info.U_FI = max(u_mat')*ppi; %full information
info.U_NI = max(ppi'*u_mat); %no information

% posterior beliefs after information shock about workers arrives:
post_w=zeros(I,0);
dpost_w=[];
for (k=1:length(dpost_eff))
    for (w=1:N)
        lwk=sum(post_eff(states(:,2)==w,k)); %likelihood of worker w conditional on posterior k
        dpost_w(end+1)=dpost_eff(k)*lwk;     %likelihood of worker w and posterior k
        pwk=post_eff(:,k); pwk(states(:,2)~=w)=0;
        post_w(:,end+1)=pwk/sum(pwk);                    %new posterior
    end
end
assert(min(sum(post_w,1))-1>-1e-9 && max(sum(post_w,1))-1<1e-9,'Coding error: All posteriors must sum to one.');
info.U_w= max(u_mat'*post_w)*dpost_w';

% posterior beliefs after information shock about tasks arrives:
post_t=zeros(I,0);
dpost_t=[];
for (k=1:length(dpost_eff))
    for (t=1:2)
        ltk=sum(post_eff(states(:,1)==t,k)); %likelihood of task t conditional on posterior k
        dpost_t(end+1)=dpost_eff(k)*ltk;     %likelihood of task t and posterior k
        ptk=post_eff(:,k); ptk(states(:,1)~=t)=0;
        post_t(:,end+1)=ptk/sum(ptk);        %new posterior
    end
end
assert(min(sum(post_t,1))-1>-1e-9 && max(sum(post_t,1))-1<1e-9,'Coding error: All posteriors must sum to one.');
info.U_t= max(u_mat'*post_t)*dpost_t';

% save setup in struct
if nargout>4
    setup.Phi = Phi;
    setup.phi_mat = phi_mat;
    setup.u_mat=u_mat;
    setup.states=states;
    setup.ppi=ppi;
    setup.actions=actions;
    setup.actionlabels=actionlbls;
    setup.employed=employed;
    setup.majority=majority;
    setup.productive=productive;
    setup.beneficial=beneficial;
end
 
H=@(q) q(q>0)'*log(q(q>0));
information=@(post,dpost,pre,dpre) ...
   dpost*arrayfun(@(k) H(post(:,k)),1:length(dpost))' - dpre*arrayfun(@(k) H(pre(:,k)),1:length(dpre))';

info.MI  =information(post_eff,dpost_eff,ppi,[1]);
info.FI=information(eye(2*N),ppi',ppi,[1]);
info.MI_t=information(post_t,dpost_t,ppi,[1]);
info.MI_w=information(post_w,dpost_w,ppi,[1]);
% M=zeros(I,2); for (t=1:2) M(states(:,1)==t,t)=1; end
% pre_t=diag(ppi)*M./sum(diag(ppi)*M,1); %posterior if tasks are revealed ex ante
% info.foc_w=information(post_t,dpost_t,pre_t,[p 1-p]); % focus on workers
% % (MI if tasks are either revealed ex-ante or ex-post... so tasks are known either way)
% M=zeros(I,N); for (w=1:N) M(states(:,2)==w,w)=1; end
% pre_w=diag(ppi)*M./sum(diag(ppi)*M,1); %posterior if workers are revealed ex ante
% info.foc_t=information(post_w,dpost_w,pre_w,q'); % focus on tasks
info.output_E = sum(prob.joint.*phi_mat,"all");
info.output_SD = sqrt(sum(prob.joint.*(phi_mat - info.output_E).^2,"all"));
info.output_H = -H(arrayfun(@(k) sum(prob.joint(productive==k)),0:8)');

end

