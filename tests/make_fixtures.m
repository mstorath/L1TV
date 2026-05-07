function make_fixtures()
% Generate MATLAB-side reference outputs for the L1TV Rust-port parity test.
% Run from the L1TV repo root with the L1TV folders on path:
%
%   matlab -batch "addpath('.', 'Auxiliary'); cd tests; make_fixtures"
%
% Writes 16 fixture files into tests/matlab_fixtures/ — 8 real + 8 circ
% covering tiny / small / medium / large N, with and without weights, plus
% alpha=0, alpha very large, integer-valued y (real), and ±pi branch-cut
% inputs (circ).
%
% The Python parity test loads each fixture and compares the L1-TV objective
% value of the Rust-port output to the MATLAB reference cost stored here.
% Pointwise x parity is intentionally not used (tied minima).

fixtureDir = fullfile(fileparts(mfilename('fullpath')), 'matlab_fixtures');
if ~exist(fixtureDir, 'dir')
    mkdir(fixtureDir);
end

% --- real-valued cases ---
addCase(fixtureDir, 'real_tiny',     [-1; 0; 1],                       1.0,  [], 'real');
addCase(fixtureDir, 'real_small',    [-2; -1; 0; 1; 2],                 0.5,  [], 'real');
rng(1);
addCase(fixtureDir, 'real_n100',     randn(100, 1),                      0.5,  [], 'real');
rng(2);
y = randn(100, 1); w = 0.5 + rand(100, 1);
addCase(fixtureDir, 'real_n100_w',   y,                                  0.5,  w, 'real');
rng(3);
addCase(fixtureDir, 'real_n2000',    randn(2000, 1),                     10.0, [], 'real');
rng(4);
addCase(fixtureDir, 'real_alpha0',   randn(50, 1),                       0.0,  [], 'real');
rng(5);
addCase(fixtureDir, 'real_huge',     randn(50, 1),                       1e6,  [], 'real');
rng(6);
addCase(fixtureDir, 'real_int',      double(randi([1, 5], 100, 1)),       1.5,  [], 'real');

% --- circle-valued cases ---
addCase(fixtureDir, 'circ_tiny',      [-pi/2; 0; pi/2],                          0.5, [], 'circ');
addCase(fixtureDir, 'circ_branchcut', [-pi+0.05; pi-0.05; -pi+0.1; -pi+0.02; 0], 0.6, [], 'circ');
rng(11);
addCase(fixtureDir, 'circ_n100',      wrapAngle(2*randn(100, 1)),               0.5, [], 'circ');
rng(12);
y = wrapAngle(2*randn(100, 1)); w = 0.5 + rand(100, 1);
addCase(fixtureDir, 'circ_n100_w',    y,                                         0.5, w,  'circ');
rng(13);
addCase(fixtureDir, 'circ_n2000',     wrapAngle(2*randn(2000, 1)),              5.0, [], 'circ');
rng(14);
addCase(fixtureDir, 'circ_alpha0',    wrapAngle(2*randn(50, 1)),                0.0, [], 'circ');
rng(15);
addCase(fixtureDir, 'circ_huge',      wrapAngle(2*randn(50, 1)),                1e6, [], 'circ');
addCase(fixtureDir, 'circ_corners',   [0; pi; -pi/2; pi/2; -pi+1e-10],          0.5, [], 'circ');

fprintf('Generated 16 fixtures in %s\n', fixtureDir);

end


function addCase(fixtureDir, name, y, alpha, w, kind)
% Run the appropriate MATLAB solver, compute the objective, save .mat.

if isempty(w)
    switch kind
        case 'real', x = L1TV_Real(y, alpha);
        case 'circ', x = L1TV_Circ(y, alpha);
    end
    wuse = ones(size(y));
else
    switch kind
        case 'real', x = L1TV_Real(y, alpha, 'weights', w);
        case 'circ', x = L1TV_Circ(y, alpha, 'weights', w);
    end
    wuse = w;
end

switch kind
    case 'real'
        cost = alpha * sum(abs(diff(x))) + sum(wuse .* abs(x - y));
    case 'circ'
        d = min(abs(x(2:end) - x(1:end-1)), 2*pi - abs(x(2:end) - x(1:end-1)));
        df = min(abs(x - y), 2*pi - abs(x - y));
        cost = alpha * sum(d) + sum(wuse .* df);
end

outFile = fullfile(fixtureDir, [name '.mat']);
save(outFile, 'y', 'alpha', 'w', 'x', 'cost', 'kind');
fprintf('  %s: N=%d alpha=%.4g cost=%.6g\n', name, numel(y), alpha, cost);

end
