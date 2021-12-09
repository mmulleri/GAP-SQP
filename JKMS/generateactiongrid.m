function grid=generateactiongrid(ticks)
grid=generategrid(ticks,1);
grid(:,1)=1-grid(:,2)-grid(:,3);
end
