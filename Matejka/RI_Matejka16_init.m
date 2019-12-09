% RI_Matejka16_init.m
%-------------------------------------
% Initialize folders and path.
%-------------------------------------

clear;

%% Add path

addpath('..');

%% Folders

% Numerical output
if ~exist('Matejka16_output', 'dir')
       mkdir('Matejka16_output')
end

% Figures
if ~exist('Matejka16_figs', 'dir')
       mkdir('Matejka16_figs')
end

%eof