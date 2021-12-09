function norm_sol=RI_portfolio_LQG(combo,ncores,Nruns)
maxNumCompThreads(ncores);
addpath('..');
load(sprintf('portfolio_input/%s_setup.mat',combo),'alpha','lambda','s_y2','s_z2','mu_y_risky','mu_y_safe','u');
load(sprintf('portfolio_input/%s_NI_solution.mat',combo),'actiongrid')
s_y=sqrt(s_y2);

% optimization vector is
% [mean(th),stdev(th),corr(th1,th2),corr(thi,yi),corr(thi,yj)]
% in that order
M = [alpha^2*s_z2, 0, -alpha, 0;
    0, alpha^2*s_z2, 0, -alpha;
    -alpha,         0, 0,     0;
    0,         -alpha, 0,     0];
m = alpha*(mu_y_risky-mu_y_safe)*[1 1 0 0]';
cov_th = @(x)  [x(2)^2,         x(3)*x(2)*x(2);
                x(3)*x(2)*x(2), x(2)^2];
cov_y = s_y2*eye(2);
cov_thy = @(x) [x(4)*x(2)*s_y,  x(5)*x(2)*s_y;
                x(5)*x(2)*s_y,  x(4)*x(2)*s_y];
Sigma = @(x) [cov_th(x),      cov_thy(x);
              cov_thy(x)',    cov_y];
means_th = @(x) [x(1); x(1)];
Q = @(x) (Sigma(x)^(-1)-M)^(-1);
x_hat = @(x) [means_th(x); 0; 0];

init_guess = [actiongrid(2),0.0001,0,0,0]'; %start near the no-information solution

llb = [-Inf,0.0000000001,-1,-1,-1]';
uub = [Inf,Inf,1,1,1]';

MI = @(x) .5*log( det(cov_th(x))*det(cov_y) / det(Sigma(x)) );

c0 = -alpha*mu_y_safe;
c1 = @(x) -1/2*x_hat(x)'*Sigma(x)^(-1)*x_hat(x);
v2 = @(x) m-Sigma(x)^(-1)*x_hat(x);
c2 = @(x) 1/2*v2(x)'*Q(x)*v2(x);
negW = @(x) det(eye(4)-M*Sigma(x))^(-1/2)*exp(c0+c1(x)+c2(x))+lambda*MI(x); 

ooptions = optimset('Display','iter','TolX',1e-12,'TolFun',1e-12,'MaxFunEvals',1e7,'MaxIter',1e7);
norm_sol = fmincon(negW,init_guess,[],[],[],[],llb,uub,[],ooptions);

mean_ttheta=means_th(norm_sol)
cov_ttheta=cov_th(norm_sol)
cov_tthetay=cov_thy(norm_sol)
u_LQG_cf=-negW(norm_sol)
I_LQG=MI(norm_sol)

x=norm_sol;

U_obj = @(ttheta1,ttheta2,y1,y2) u([1-ttheta1-ttheta2, ttheta1, ttheta2],[mu_y_safe mu_y_risky+y1 mu_y_risky+y2]);

EU_vec = ones(Nruns,1);
Ndraws = 1e7;
fprintf('Computing the objective value with Monte-Carlo methods.\n Executing %i runs with %g draws each.\n',Nruns,Ndraws);
Sig_LQG=Sigma(norm_sol)
parfor t=1:Nruns
    rng(t);
    iinputs = mvnrnd([mean_ttheta;0;0]',Sig_LQG,Ndraws);

    EU_vec(t) = sum(arrayfun(U_obj,iinputs(:,1),iinputs(:,2),iinputs(:,3),iinputs(:,4)))/Ndraws ...
        - lambda*MI(norm_sol);
    %net utility (after information costs)
end
u_LQG = mean(EU_vec)
u_LQG_stderr = std(EU_vec);
u_LQG_95perc_CI = [u_LQG-1.96*u_LQG_stderr,u_LQG+1.96*u_LQG_stderr];


save(sprintf('portfolio_input/%s_LQG.mat',combo),'mean_ttheta','cov_ttheta','u_LQG','u_LQG_cf','Sig_LQG','I_LQG');
end
