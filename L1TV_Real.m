function x = L1TV_Real( y, alpha, varargin )
%L1TV_Real Exact solver for the univariate L1TV problem with real-valued
%data. i.e. it computes the solution of
% min_{x \in R^n} \alpha sum_{i=1}^{n-1} |x_i - x_{i+1}| + sum_{i=1}^{n} w_i |x_i - y_{i}| 
% where R is the real numbers 
%
% Input:
% y: n-vector of real numbers
% alpha: regularization parameter
% Optional arguments
% 'weights': pointwise weights for the data fidelity (n-vector of positve numbers)
%
% Reference:
% Storath, M., Weinmann, A., & Unser, M. (2016). 
% Exact algorithms for L^1-TV regularization of real-valued or circle-valued signals. 
% SIAM Journal on Scientific Computing, 38(1), A614-A630.

% parse input
ip = inputParser;
ip.addParameter('useDistTrans', true);
ip.addParameter('weights', ones(size(y)));

parse(ip, varargin{:});
par = ip.Results;

% initialization
uniqueValues = unique(y); % determine unique angles
V = sort(uniqueValues); % sort values
N = numel(y);
K = numel(V);
B = zeros(K,N); % tabulation
pen = zeros(K,1);

% tabulation
B(:,1) = par.weights(1) .* abs(V - y(1));
for n=2:N
    d = par.weights(n) .* abs(V - y(n));
    if par.useDistTrans % Using distance transforms, the complexity is O(NK)
        pen = distTransformL1(B(:,n-1), V, alpha);
    else % The naive implemention is O(NK^2)
        for k=1:K
            pen(k) = min( B(:,n-1) +  alpha * abs(V - V(k)));
        end
    end
    B(:,n) = d + pen;
end

% backtracking
[~,l] = min(B(:,N));
x(N,1) = V(l);
for n=N-1:-1:1
    [~, l] = min(B(:, n) + alpha * abs(V - x(n+1)));
    x(n) = V(l);
end


end

