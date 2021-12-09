function RI_portfolio_sims(combo)
load(sprintf('portfolio_input/%s_setup.mat',combo),'u','stategrid','xrange_FI','prior','lambda','alpha','s_z2','s_y2','mu_y_risky','mu_y_safe');

switch combo
    case 'A'
        actiongrid_sims=[0, 20.727, 20.727; 0,14.418, 1.429; 0,1.429, 14.418];
        p_marg_sims=[0.066;0.467;0.467];
    case 'B'
        actiongrid_sims=[0, -23.006, -23.006; 0,55.214, 10.431; 0,10.431, 55.214];
        p_marg_sims=[0.185;0.408;0.408];
    case 'C'
        actiongrid_sims=[0, -77.975, -77.975; 0,45.748, -0.091; 0,-0.091, 45.748];
        p_marg_sims=[0.024;0.488;0.488];
    case 'D'
        actiongrid_sims=[0,19.328,19.328;0,19.890,-5.666;0,-5.666,19.890;0,-8.737,-8.737];
        p_marg_sims=[0.361;0.248;0.248;0.142];
end
actiongrid_sims(:,1)=1-actiongrid_sims(:,2)-actiongrid_sims(:,3);
p_marg_sims=p_marg_sims/sum(p_marg_sims);
u_mat_sims   = u(actiongrid_sims,stategrid);
b_logscales_sims = -max(u_mat_sims,[],2)/lambda;
b_mat_sims = exp(u_mat_sims/lambda + b_logscales_sims);
b_sims       = b_mat_sims*p_marg_sims;
[~,p_joint_sims,u_sims,I_sims] = GAP_components(p_marg_sims,u_mat_sims,lambda,prior);
save(sprintf('portfolio_input/%s_sims.mat',combo),'p_joint_sims','u_sims','I_sims','p_marg_sims','actiongrid_sims','u_mat_sims')
end