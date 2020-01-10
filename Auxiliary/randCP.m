function r = randCP( r, lambda)
%randCP Generates a random vector distributed according to a certain 
% compound Poisson distribution with parameter lambda

x = rand(size(r)); % uniform distr. vector
r(x(:) <= exp(-lambda)) = 0; % compound Poisson

end

