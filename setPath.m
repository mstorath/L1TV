% sets the paths 
disp('Setting Matlab path...');
folder = fileparts(which(mfilename));
addpath(...
    fullfile(folder, ''),...
    fullfile(folder, 'Auxiliary') ...
    );

% save pathdef
savepath;