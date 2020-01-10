function d = distAngle( phi, psi )
%distAngle Distance between two angles phi and psi

if (all(abs(phi) <= pi)) && (all(abs(psi) <= pi))
    % cheap computation for angles in the interval [-pi, pi]
    aux = phi - psi;
    d = min(abs([aux(:) + 2 * pi, aux(:), aux(:) - 2 * pi]), [], 2);
    d = reshape(d, size(aux));
else
    % more expensive calculation for angles outside the interval [-pi, pi]
    aux = abs(angle(exp(1i * (phi - psi))));
    d = min(aux, 2*pi-aux);
end

