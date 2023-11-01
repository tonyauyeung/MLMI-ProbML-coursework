function GPplot3d(mu, s2, x, y, xs1, xs2, num_split_obs, num_split_grid, name)
    figure()
    subplot(2, 3, [1, 2, 4, 5])
    mesh(xs1, xs2, reshape(mu, num_split_grid, num_split_grid))
    hold on;
    % mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11))
    plot3(reshape(x(:,1),num_split_obs,num_split_obs), ...
        reshape(x(:,2),num_split_obs,num_split_obs), ...
        reshape(y,num_split_obs,num_split_obs), '+')
    title(name)

    xs1_margin = xs1(1, :)';
    xs2_margin = xs2(:, 1);
    mu = reshape(mu, num_split_grid, num_split_grid);
    s2 = reshape(s2, num_split_grid, num_split_grid);
    mu_xs1_margin = mu(1, :)';
    mu_xs2_margin = mu(:, 1);
    s2_xs1_margin = s2(1, :)';
    s2_xs2_margin = s2(:, 1);
    subplot(2, 3, 3)
    GPplot(mu_xs1_margin, s2_xs1_margin, x(1:num_split_obs:length(x), 1), ...
        y(1:num_split_obs:length(y)), xs1_margin, ...
        'Marginal error bar, with fixed x_2 = -3')
    subplot(2, 3, 6)
    GPplot(mu_xs2_margin, s2_xs2_margin, x(1:num_split_obs, 2), ...
        y(1:num_split_obs), xs2_margin, ...
        'Marginal error bar, with fixed x_1 = -3')
    % f_xs1_margin = [mu_xs1_margin + 2 * sqrt(s2_xs1_margin); 
    %     flipdim(mu_xs1_margin - 2 * sqrt(s2_xs1_margin), 1)];
    % figure();
    % fill([xs1_margin; flipdim(xs1_margin,1)], f_xs1_margin, [7 7 7]/8)
    % hold on; plot(xs1_margin, mu_xs1_margin); plot(x(1:11:121, 1), y(1:11:121), '+')
    % title('Marginal error bar, with fixed x_2 = -3')
end