%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example script for Dynamic Correlation Toolbox
%
% File created by Martin Lindquist 07/22/14
%
% Makes use of functions from the UCSD_Garch toolbox by Kevin Shepard (Please see license agreement)
%
% Before running this script, begin by adding the DC_toolbox and all its subdirectories to the Matlab path.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create simulated data set 

% Define data dimensions
p = 10;         % Number of nodes
T = 128;        % Numer of time points

% Generate null data
mu = zeros(p,1);
Sigma = eye(p);
dat=mvnrnd(mu,Sigma,T);     

% Note the input data has dimensions T-by-p (time by #nodes)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit DCC
[Ct1, Ht1] = DCC(dat);

% Ct1 is the dynamic correlation matrix 
% Ht1 is the dyanamic covariance matrix


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fit sliding-window correlations
windowsize = 20;
[ Ct2 ] = sliding_window(dat,windowsize);

% Ct2 is the sliding window correlation matrix 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot some of the results

figure
subplot 221
imagesc(Ct1(:,:,100), [-1 1])          % Plot the conditional correlation matrix at time 100
colorbar
title('DCC - conditional correlation at time 100')

subplot 222
plot(squeeze(Ct1(1,3,:)))    % Plot the dyanamic correlation between nodes 1 and 3.
ylim([-0.7 0.7])
hold
plot(1:T,zeros(T,1),'-r')    % Plot the true dyanamic correlation between nodes 1 and 3.
title('DCC - dynamic correlation between nodes 1 and 3')


subplot 223
imagesc(Ct2(:,:,100), [-1 1])          % Plot the conditional correlation matrix at time 100
colorbar
title('SWC - conditional correlation at time 10')

subplot 224
plot(squeeze(Ct2(1,3,:)))    % Plot the dyanamic correlation between nodes 1 and 3.
ylim([-0.7 0.7])
hold
plot(1:T,zeros(T,1),'-r')    % Plot the true dyanamic correlation between nodes 1 and 3.
title('SWC - dynamic correlation between nodes 1 and 3')

