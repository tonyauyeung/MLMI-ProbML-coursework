function f = GPplot(mu, s2, x, y, xs, plot_name)
    f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)];
    fill([xs; flipdim(xs,1)], f, [7 7 7]/8)
    hold on; plot(xs, mu); 
    plot(x, y, '+')
    title(plot_name)
end