function [ttimes,flags,obj_vs] = RI_timed_runs(u_mat,b_mat,ppi,llambda,conv_tol_IE,neg_w)
ttimes=zeros(1,2);
flags=zeros(1,2);
obj_vs=zeros(1,2);

% SQP ALGORITHM
runf = @() GAP_SQP(u_mat,ppi,llambda,...
    'display','off',...
    'conv_tol_IE',conv_tol_IE,...
    'save_hist', false);
[p_marg,~,~,exitflag_A] = runf();
ttime_A = timeit(runf,4);

b = b_mat*p_marg;
p_marg = max(p_marg,0);
p_marg = p_marg/sum(p_marg);
obj_A = - neg_w(p_marg);

% BA ALTERNATIVE
runf = @() solve_BA(u_mat,ppi,llambda,...
    'display','off',...
    'stopping_rule','objective',...
    'save_hist', false);
[p_marg,~,~,exitflag_B] = runf();
ttime_B = timeit(runf,4);

p_marg = max(p_marg,0);
p_marg = p_marg/sum(p_marg);
b_X = b_mat*p_marg;
obj_B = -neg_w(p_marg);

ttimes = [ttime_A ttime_B]
flags  = [exitflag_A exitflag_B];
obj_vs = [obj_A obj_B];

end