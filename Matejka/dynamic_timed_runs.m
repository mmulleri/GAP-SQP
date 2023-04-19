function [stats,comparison] = dynamic_timed_runs(u_mat,mmu_init,ppi_trans,llambda,bbeta,T)
[LA,LS] = size(u_mat);

%% GAP VARIANT
fprintf('Running GAP for output.\n');
tGAP=tic;
[pmarg_GAP,Pcond_GAP,prior_GAP,mu_GAP,mu_hist_GAP,v_GAP,b_GAP,stats_GAP] = dynamic_matejka(u_mat,mmu_init,ppi_trans,llambda,bbeta,T,true,true);
ttime_GAP=toc(tGAP);

%% MX BA VARIANT
fprintf('Running MX for output.\n');
tBA=tic;
[pmarg_BA,Pcond_BA,prior_BA,mu_BA,mu_hist_BA,v_BA,b_BA,stats_BA] = dynamic_matejka(u_mat,mmu_init,ppi_trans,llambda,bbeta,T,true,false);
ttime_BA=toc(tBA);

stats=[stats_GAP;stats_BA];

%% compare solutions
Delta_joint = max(mu_GAP-mu_BA,[],"all");

comparison = [d_IE(prior_GAP,b_GAP,b_BA),d_IE_onpath(prior_GAP,b_GAP,b_BA,mu_hist_GAP),d_CDF(prior_GAP,prior_BA)]; 
fprintf('Comparing solutions: largest IE distance in any history %6f, on-path IE distance %6f, largest belief distance in any history %6f.\n\n',comparison);



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
end