function x = L1TV_Circ( y, alpha, varargin )
%L1TV_Circ Exact solver for the univariate L1TV problem with circle valued
%data. i.e. it computes the solution of
% min_{x \in T^n} \alpha sum_{i=1}^{n-1} d(x_i, x_{i+1}) + sum_{i=1}^{n} w_i d(x_i, y_{i}) 
% where T is the unit circle (torus) and d(x, y) is the shortest arc
% distance between x and y
%
% Input:
% y: n-vector of phase angles between [-pi, pi]
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
yCom = exp(1i * y); % complex number representation of y
yAng = angle(yCom); % assure angle to be in [-pi, pi]
uniqueValues = unique(yAng); % determine uniqe angles
V = [uniqueValues; angle( -exp(1i * uniqueValues))]; % add antipodal points
V = sort(V); % sort candidate values
N = numel(yAng);
K = numel(V);
B = zeros(K,N); 
pen = zeros(K,1);


% compute tabulation
B(:,1) = par.weights(1) .* distAngle(V, yAng(1));
if par.useDistTrans % Using distance transforms, the complexity is O(N K)
    % auxiliary structure for distance transforms
    Vrep = cat(1, V-2*pi, V, V+2*pi);
    for n=2:N
        d = par.weights(n) .* distAngle(V, yAng(n));
        Brep = repmat(B(:,n-1), [3,1]);
        aux = distTransformL1(Brep, Vrep, alpha);
        pen = min(reshape(aux, K, 3), [], 2);
        B(:,n) = d + pen;
    end
else
    for n=2:N % Naive implementation is O(N K^2)
        d = par.weights(n) .* distAngle(V, yAng(n));
        for k=1:K
            pen(k) = min( B(:,n-1) +  alpha * distAngle(V, V(k)));
        end
        B(:,n) = d + pen;
    end
    
end

% backtracking
[~,l] = min(B(:,N));
x(N,1) = V(l);
for n=N-1:-1:1
    [~, l] = min(B(:, n) + alpha * distAngle(V, x(n+1)));
    x(n) = V(l);
end


end

