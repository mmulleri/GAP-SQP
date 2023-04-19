function [pmarg,Pcond,pprior,mu,mu_hist,v,b,stats] = dynamic_matejka(u,mmu_init,ppi_trans,llambda,bbeta,T,verbatim,GAP)

% This function solves the problem using the Miao-Xing (2020) algorithm.
% If GAP is false, value functions are updated using BA algorithm.
% Otherwise, GAP-SQP is used.

%% INITIALIZATIONS
t_0=tic;
[LA LS] = size(u);
converged = false;
conv_tol_IE=1e-6;

% Value functions
% b(:|a_old,t) is the optimal attention vector for the continuation problem.
% v(:,a|t) captures the present + future payoffs of action a
b = ones(LS,LA,T+1);
v = u .* ones(1,1,T);

% Choice probabilities:
%  Conditionals Pcond(a | s, a_old, t)
%     initialized to full-information solution
Pcond=zeros(LA,LS,LA,T);
for is=1:LS
    [~,amax] = max(u(:,is));
    Pcond(amax,is,:,:)=1;
end

% beliefs:
%  pprior(s | a_old,t) pprior in period after a_old
%    (initialized to unconditional beliefs)
pprior = mmu_init*ones(1,LA).*ones(1,1,T);
%  mu(s,a_old,t) joint probabilities
%  mu_hist(a_old|t) overall probability of history a_old
[mu,mu_hist,pprior] = belief_update(Pcond,pprior);

%  Initial values from straightforward BA updating:
%   pmarg(a | a_old,t) initialized via from conditionals.
[pmarg,Pcond,b,v]  = value_update_BA(pprior,Pcond);

ddist = 1.0;

ccounter = 0;

% [w,MI1,MIlast] = computeWelfare(Pcond,pmarg,pprior,b,v);
% methods={'BA','GAP'};
stats = zeros(0,11); %[GAP,toc(t_0),ccounter,w,MI1,MIlast,checkSufficiency(pprior,v,b),NaN,NaN,NaN,NaN];
% fprintf('%3s | %6.0f s: round %6i, welfare %9e, MI %3f -> %3f, sufficiency %6e, IE dist %6e, average IE dist on path %6e, belief dist %6e, obj dist %6e.\n',methods{GAP+1},stats(end,2:end));

       
dt = 6; %denote performance every dt seconds;

b_outdated=b;
prior_outdated=pprior;
pmarg_outdated=pmarg;
Pcond_outdated=Pcond;

while (~converged)
    ccounter = ccounter + 1;


    % First we scroll forward to obtain probabilities and beliefs
    [mu,mu_hist,pprior] = belief_update(Pcond,pprior);
    
    % Then we scroll back to obtain continuation values and choices...
    if GAP
        [pmarg,Pcond,b,v]  = value_update_GAP(pprior,pmarg,.1*conv_tol_IE);
    else
        [pmarg,Pcond,b,v]  = value_update_BA(pprior,Pcond);
    end
    
    converged = check_convergence(pprior,prior_outdated,b,b_outdated,t_0);

    if verbatim & (toc(t_0)>=dt*size(stats,1) | converged)
        [w,MI1,MIlast] = computeWelfare(Pcond,pmarg,pprior,b,v);
        methods={'BA','GAP'};
        stats(end+1,:) = [GAP,toc(t_0),ccounter,w,MI1,MIlast,checkSufficiency(pprior,v,b),d_IE(pprior,b,b_outdated),d_IE_onpath(pprior,b,b_outdated,mu_hist),d_CDF(pprior,prior_outdated),d_welfare(pprior,prior_outdated,b,b_outdated)];
        fprintf('%3s | %6.0f s: round %6i, welfare %9e, MI %3f -> %3f, sufficiency %6e, IE dist %6e, average IE dist on path %6e, belief dist %6e, obj dist %6e.\n',methods{GAP+1},stats(end,2:end));

        b_outdated=b;
        prior_outdated=pprior;
        pmarg_outdated=pmarg;
        Pcond_outdated=Pcond;

    end

    if converged & GAP
        % update one more time without perturbation
        value_update_GAP(pprior,pmarg,0);
    end
end

    function [mu_joint,mu_h,priors] = belief_update(Pcond,prior_candidate)
        % Initialize at t=1;
        mu_joint(:,:,1) = 1/LA*mmu_init*ones(1,LA);
        priors=prior_candidate;
        for t=1:T
            mu_joint(:,:,t+1)=sum(reshape(ppi_trans(:,:),[LS 1 LS]).*sum(reshape(Pcond(:,:,:,t),[1 LA LS LA]) .* reshape(mu_joint(:,:,t),[1 1 LS LA]),4), 3);
            % dimensions denote [s_new, a, s, a_old]

            % infer on-path beliefs from choices
            mu_h(:,t)=sum(mu_joint(:,:,t),1)';
            onpath=(mu_h(:,t)>1/LA*1e-3*conv_tol_IE);
            priors(:,onpath,t)=mu_joint(:,onpath,t)./reshape(mu_h(onpath,t),[1 sum(onpath)]);
        end
    end

    function [pmarg,Pcond,b_updated,v_updated]=value_update_GAP(pprior,pmarg_guess,epsilon)
        b_updated(:,:,T+1)=ones(LS,LA);
        for t=T:-1:1
            v_updated(:,:,t)=u'+bbeta*llambda*sum(reshape(ppi_trans',[LS 1 LS]).*log(reshape(b_updated(:,:,t+1)',[1 LA LS])),3);
            % reshaped to dimensions [s, a, s_new]

            parfor a_old=1:LA
                [marg,~,~,exitflag]= GAP_SQP(v_updated(:,:,t),pprior(:,a_old,t),llambda,'Display','none','initial_p',pmarg_guess(:,a_old,t),'MaxIterations',1,'MaxLinIt',0); %'conv_tol_IE',.1*conv_tol_IE);
                % Start from last iteration as guess for marginal
                bstar=sum(reshape(marg,[1 LA]).* exp(v_updated(:,:,t)/llambda),2);
                b_updated(:,a_old,t)=bstar;

                pmarg(:,a_old,t)=marg;
                % compute conditionals with full support from perturbed marginals:
                marg=(1-epsilon)*marg+epsilon*1/LA*ones(LA,1);
                Pcond(:,:,a_old,t)=reshape(marg,[LA 1]) .* exp(v_updated(:,:,t)'/llambda) ./ sum(reshape(marg,[LA 1]).* exp(v_updated(:,:,t)'/llambda),1);%reshape(bstar,[1 LS]);
            end
        end
    end

    function [pmarg,Pcond,b_updated,v_updated]=value_update_BA(pprior,Pcond_guess)
        b_updated(:,:,T+1)=ones(LS,LA);
        for t=T:-1:1
            v_updated(:,:,t)=u'+bbeta*llambda*sum(reshape(ppi_trans',[LS 1 LS]).*log(reshape(b_updated(:,:,t+1)',[1 LA LS])),3);
            % reshaped to dimensions [s, a, s_new]
            pmarg(:,:,t)=reshape(sum(Pcond_guess(:,:,:,t) .* reshape(pprior(:,:,t),[1 LS LA]),2),[LA LA]);
            % dimensions denote [a, s, a_old]
            b_updated(:,:,t)=reshape(sum(reshape(pmarg(:,:,t),[1 LA LA]).*exp(reshape(v_updated(:,:,t),[LS LA 1])/llambda),2), [LS LA]);
            % dimensions denote [s, a, a_old]
            Pcond(:,:,:,t)=reshape(pmarg(:,:,t),[LA 1 LA]) .* exp(reshape(v_updated(:,:,t)',[LA LS 1])/llambda) ./ reshape(b_updated(:,:,t),[1 LS LA]);
            % dimensions denote [a, s, a_old]
        end
    end

    function [maxscore] = checkSufficiency(pprior,v,b)
        % % Check for sufficiency; should be <= 0 everywhere
        suff_mat = reshape(sum(reshape(pprior,[LS 1 LA T]) .* exp(reshape(v,[LS LA 1 T])/llambda) ./ reshape(b(:,:,1:T),[LS 1 LA T]) ,1),[LA LA T]) - 1;
        % reshaped to dimensions [s,a,a_old,t]
        maxscore = max(suff_mat,[],"all");
    end

    function [welfare,MI1,MIlast] = computeWelfare(Pcond,pmarg,pprior,b,v)
        [~,mu_h,pprior] = belief_update(Pcond,pprior);
        % determine information acquisition in each period
        MI_entries=reshape(pprior,[1 LS LA T]) .* Pcond .* log( Pcond ./ reshape(pmarg,[LA 1 LA T]) );
        MI_entries(reshape(pmarg<1e-12,[LA 1 LA T]) & true(1,LS))=0;
        MI=reshape(sum(sum(MI_entries,2),1),[LA T]);
        MI_across_periods=sum(mu_h .* MI,1);
        MI1    =MI_across_periods(1);
        MIlast =MI_across_periods(end);
        % % dimensions denote [a, s, a_old, t]
        cons_u = reshape(sum(sum(reshape(pprior,[1 LS LA T]) .* Pcond .* u,2),1),[LA T]);
        welfare = sum(reshape(bbeta.^(1:T),[1 T]) .* mu_h .* (cons_u - llambda * MI),"all");
    end

    function ddist = d_IE(pprior,b,b_outdated)    % attention vector distance
        ddist = max(sqrt(sum(llambda* pprior.*(log(b_outdated(:,:,1:T))-log(b(:,:,1:T))).^2,1) ),[],"all");
        % reshaped to dimensions [s,a,t]
    end

    function ddist = d_IE_onpath(pprior,b,b_outdated,mu_h)    % attention vector distance
        ddist = sum(reshape(mu_h,[1 LA T]) .* sqrt(sum(llambda* pprior.*(log(b_outdated(:,:,1:T))-log(b(:,:,1:T))).^2,1) ),"all");
        % reshaped to dimensions [s,a,t]
    end

    function ddist = d_CDF(pprior,prior_outdated) % belief distance between CDFs
        % compute the absolute difference in CDF for each (a,t) combo
        dCDF = abs(pagemtimes(tril(ones(LS)),pprior-prior_outdated));
        % the distance is the maximum weighted average across all a,t pairs
        ddist = max(sum(1/LS.*dCDF,1),[],"all");
    end

    function ddist = d_welfare(pprior,prior_outdated,b,b_outdated) % welfare distance
        wh = sum(llambda*pprior.*log(b(:,:,1:T)),1);
        wh_outdated = sum(llambda*prior_outdated.*log(b_outdated(:,:,1:T)),1);
        ddist = max(abs(wh-wh_outdated),[],"all");
    end

    function converged = check_convergence(pprior,prior_outdated,b,b_outdated,timer)
        % converged = max([d_IE(pprior,b,b_outdated),d_CDF(pprior,prior_outdated)])<conv_tol_IE;
        converged = toc(timer)>100;
    end


end