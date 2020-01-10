function y = randl( varargin )
%randl Random normal Laplacian noise, mean = 0, variance sigma^2 = 1
% pdf: $\frac{1}{\sqrt{2}} e^{- \sqrt{2} |x|}$
% Same input arguments as function rand

x = rand( varargin{:} ) - 0.5;
y = -sign(x) .* log(1 - 2 * abs(x)) / sqrt(2);

end

