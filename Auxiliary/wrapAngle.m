function x = wrapAngle( y )
%wrapAngle Wraps the angle y to the interval [-pi, pi]

x = angle(exp(1i * y));

end

