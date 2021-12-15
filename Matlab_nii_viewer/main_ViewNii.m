%%main_ViewNii
%% Clear environemnt
clear all;
close all;
clc;

addpath('NIfTI');
%% Select nii file
[filename,pathname] = uigetfile('*.nii', 'Pick a NIFTI file');

%% Load file
filepath = [pathname,filename];
nii = load_nii(filepath);

%% Open GUI
view_nii(nii);