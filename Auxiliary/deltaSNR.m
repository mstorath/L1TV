function dsnr = deltaSNR( groundTruth, data, estimate, dataSpace)
%deltaSNR Computes the signal to noise ratio improvement of a restoration

switch dataSpace
    case 'circ'
        dist = @(x,y) distAngle(x,y);
    case 'real'
        dist = @(x,y) abs(x - y);
end

dsnr = 10 * log10(sum( dist(groundTruth, data).^2 ) / sum( dist(groundTruth, estimate).^2 ));

end

