function [grid]=generategrid(ticks,dflt)
N = length(ticks);
[a,b]=ndgrid(1:N,1:N);
grid=[dflt*ones(N*N,1),ticks(a,:),ticks(b,:)];
end