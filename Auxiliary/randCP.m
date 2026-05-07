function r = randCP( r, lambda)
%randCP Bernoulli-thinned innovations approximating a compound Poisson.
% Each entry of r is independently zeroed with probability exp(-lambda);
% otherwise it is left at the supplied value. This matches the marginal
% P(N = 0) of a Poisson(lambda) point process and is used in the demos as
% a cheap surrogate for compound-Poisson innovations (good for small
% lambda; not the actual compound Poisson distribution).

x = rand(size(r)); % uniform distr. vector
r(x(:) <= exp(-lambda)) = 0;

end

