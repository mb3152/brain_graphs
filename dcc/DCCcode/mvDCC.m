function [V] = mvDCC(Dat)
% function [Ct, Ht] = DCC(dat)

% Run multivariate GARCH model using the DCC estimator separately for each bivariate pair of nodes--call DCC separately for each pair
% 
% INPUTS:
%
%      dat          Zero mean T by p matrix
% 
% OUTPUTS:
%
%      V            p by p by T array of conditional correlations
%
%
% File created by Martin Lindquist on 04/18/15
% Last update: 04/18/15


[T,p] = size(Dat);
V = ones(p,p,T);

for j=1:(p-1),    
    for k=(j+1):p,
  
        [ Ct ] = DCC([Dat(:,j) Dat(:,k)]); 
        V(j,k,:) = Ct(1,2,:); 
        V(k,j,:) = Ct(1,2,:); 

    end
end


