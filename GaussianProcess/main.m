%% coursework setup%%
run ../../gpml-matlab-master/startup.m
data = load('cw1a.mat');
x = data.x;
y = data.y;
xs = linspace(-3,3, 100)';
% figure();
% plot(x, y, '+')

%% Question a%%
meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood
hyp = struct('mean', [], 'cov', [-1 0], 'lik', 0);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
nlz = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu_init, s2_init] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

figure()
subplot(3, 6, [1, 2, 3, 7, 8, 9])
GPplot(mu, s2, x, y, xs, '(a)optimal hyper parameters')

subplot(3, 6, [4, 5, 6, 10, 11, 12])
GPplot(mu_init, s2_init, x, y, xs, '(b)initial hyper parameters')

subplot(3, 6, [13, 14])
hyp = struct('mean', [], 'cov', [log(0.128) 0], 'lik', 0);
[mu_init, s2_init] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
GPplot(mu_init, s2_init, x, y, xs, '(c)optimal length scale')

subplot(3, 6, [15, 16])
hyp = struct('mean', [], 'cov', [-1 log(0.897)], 'lik', 0);
[mu_init, s2_init] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
GPplot(mu_init, s2_init, x, y, xs, '(d)optimal signal variance')

subplot(3, 6, [17, 18])
hyp = struct('mean', [], 'cov', [-1 0], 'lik', log(0.118));
[mu_init, s2_init] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
GPplot(mu_init, s2_init, x, y, xs, '(e)optimal noise')

% subplot(2, 1, 2)
% GPplot(mu_init, s2_init, x, y, xs, 'unfitted')
disp(-nlz)
fprintf('Optimal hyper-params: lengthscale = %d, sigma = %d, noise = %d', ...
     exp(hyp2.cov(1)), exp(hyp2.cov(2)), exp(hyp2.lik));
%% Question b%%
num_test = 5;
hyp_opts = zeros(num_test + 1, 3);
hyp_opts(1, :) = [hyp2.cov(1), hyp2.cov(2), hyp2.lik];
params = [[-4, -4, -4];[0, 0, 0]; [1, 0, 0]; [0, 1, 0]; [0, 0, 1]];
evidence = [gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y)];
for i = 1:num_test
    hyp_init = struct('mean', [], 'cov', [params(i, 1), params(i, 2)], 'lik', params(i, 3));
    hyp_opt = minimize(hyp_init, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    evidence(i + 1) = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
    hyp_opts(i + 1, :) = [hyp_opt.cov(1), hyp_opt.cov(2), hyp_opt.lik];
end
disp(hyp_opts);
disp(evidence);

figure()
[mu, s2] = gp(hyp_opt, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
subplot(2, 1, 2)
GPplot(mu, s2, x, y, xs, 'local optimum_2')
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
subplot(2, 1, 1)
GPplot(mu, s2, x, y, xs, 'local optimum_1')

%% Question c%%
covfunc = @covPeriodic;
meanfunc = [];
likfunc = @likGauss;
hyp = struct('mean', [], 'cov', [0 0 0], 'lik', 1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
nlz = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

figure()
GPplot(mu, s2, x, y, xs, 'Periodic Covariance')
fprintf(['Optimal hyper-params: lengthscale = %d, sigma = %d, p = %d, noise = %d.\n' ...
    'Log-likelihood = %d'], ...
    exp(hyp2.cov(1)), exp(hyp2.cov(3)), exp(hyp2.cov(2)), exp(hyp2.lik), -nlz);

%% Question d%%
x = linspace(-5, 5, 200)';
num_samples = 5;
covfunc = {@covProd, {@covPeriodic, @covSEiso}};
hyp = struct('mean', [], 'cov', [-0.5 0 0 2 0], 'lik', 1);
K = feval(covfunc{:}, hyp.cov, x);
L = chol(K + 1e-6 * eye(200));
f = L' * randn(200, num_samples);
figure();
t = tiledlayout(2,2,'TileSpacing','Compact','Padding','Compact');
nexttile
subplot(2, 2, [1, 2])
plot(x, f);
title('Periodic-SEiso-combined covariance')

covfunc = @covPeriodic;
hyp = struct('mean', [], 'cov', [-0.5 0 0], 'lik', 1);
K = feval(covfunc, hyp.cov, x);
L = chol(K + 1e-6 * eye(200));
f = L' * randn(200, num_samples);
subplot(2, 2, 3)
plot(x, f);
title('Periodic covariance')

covfunc = @covSEiso;
hyp = struct('mean', [], 'cov', [2 0], 'lik', 1);
K = feval(covfunc, hyp.cov, x);
L = chol(K + 1e-6 * eye(200));
f = L' * randn(200, num_samples);
subplot(2, 2, 4)
plot(x, f);
title('SEiso covariance')

%% Question e%%
data = load('cw1e.mat');
x = data.x; y = data.y;
num_split = 200;
[xs1, xs2] = meshgrid(linspace(-3, 3, num_split), linspace(-3, 3, num_split));
xs = [reshape(xs1, [], 1) reshape(xs2, [], 1)];
mesh(reshape(x(:,1),11,11),reshape(x(:,2),11,11),reshape(y,11,11))

covfunc = @covSEard;
meanfunc = [];
likfunc = @likGauss;
hyp = struct('mean', [], 'cov', [0 0 0], 'lik', 1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
nlz1 = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
GPplot3d(mu, s2, x, y, xs1, xs2, 11, num_split, ...
    ['covSEard, log-likelihood= ', num2str(-nlz1)])
K = feval(covfunc, hyp2.cov, x);
cp1 = log(det(K + exp(2 * hyp2.lik) * eye(length(K)))) / 2;
df1 = -1 / 2 * y' * inv(K + exp(2 * hyp2.lik) * eye(length(K))) * y;
h1 = [exp(hyp2.cov(1)), exp(hyp2.cov(2)), exp(hyp2.cov(3)), exp(2 * hyp2.lik)];

covfunc = {@covSum, {@covSEard, @covSEard}};
hyp.cov = 0.1 * randn(6, 1);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
nlz2 = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
GPplot3d(mu, s2, x, y, xs1, xs2, 11, num_split, ...
    ['covSEard_1 + covSEard_2, log-likelihood= ', num2str(-nlz2)])
K = feval(covfunc{:}, hyp2.cov, x);
cp2 = log(det(K + exp(2 * hyp2.lik) * eye(length(K)))) / 2;
df2 = -1 / 2 * y' * inv(K + exp(2 * hyp2.lik) * eye(length(K))) * y;
h2 = [exp(hyp2.cov(1)), exp(hyp2.cov(2)), exp(hyp2.cov(3)), exp(hyp2.cov(4)), exp(hyp2.cov(5)), exp(hyp2.cov(6)), exp(2 * hyp2.lik)];

const = length(y) / 2 * log(2 * pi);
fprintf(['df1: %d, cp1: %d\ndf2: %d, cp2: %d\nconst: %d\n hpy1: %d, ' ...
    '%d, %d, %d\nhpy2: %d, %d, %d, %d, %d, %d, %d'], df1, cp1, df2, cp2, const, ...
    h1(1), h1(2), h1(3), h1(4), h2(1), h2(2), h2(3), h2(4), h2(5), h2(6), h2(7))