function B = distTransformL1( B, R, alpha )
%distTransformL1 Fast computation of L1 distance transform

K = numel(B);
% forward pass
for k=2:K
    B(k) = min( B(k), B(k-1) + alpha * (R(k) - R(k-1)));
end
% backward pass
for k=K-1:-1:1
    B(k) = min( B(k), B(k+1) + alpha * (R(k+1) - R(k)));
end

end

