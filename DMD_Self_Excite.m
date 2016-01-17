function [mu_hat W_hat loss]=DMD_Self_Excite(data,mu_bar,tau,alpha,beta,W_min,W_max,mu_hat1,W_hat1)
%DMD_Self_Excite - Runs DMD with additive parametric dynamics  
%
% See "Online Optimization in Dynamic Environments" (arXiv 1307.5944), 
% specifically sections 7.3 and 6.2 by Eric C. Hall and Rebecca M. Willett
%
% This function uses the following dynamical model:
%
%   x_t ~ Poiss(mu_t)
%   mu_(t+1) = tau*mu + W*x_t + (1-tau)*mu_bar
%
% where x_t are the observations, mu is the time varying rate functions of 
% the system, and W is the network structure.  The function takes in the
% observations x_t and comes up with the streaming predictions of mu and W.
% 
% INPUT
%   data    - Observations of the system, each column is an observation
%   mu_bar  - Baseline intensity
%   tau     - Parameter of dynamical model
%
% OPTIONAL INPUT
%   alpha       - Intensity step size parameter, eta_t = alpha/sqrt(t)
%   beta        - Network step size parameter, rho=beta/sqrt(t)
%   W_min       - Minimum value any entry of W matrix can take
%   W_max       - Maximum value any entry of W matrix can take
%   mu_hat1     - Prediction of mu at time t=1
%   W_hat1      - Prediction of W at time t=1
%
% OUTPUT
%   mu_hat      - Predictions of mu, each column is a prediction
%   W_hat       - Final prediction of W
%   loss        - Time evolving instantaneous loss
%
% Code written: Sep 12 2014
% Written by: Eric C. Hall

if ~exist('alpha','var')||isempty(alpha)
    alpha=1;
end

if ~exist('beta','var')||isempty(beta)
    beta=1;
end

if ~exist('W_min','var')||isempty(W_min)
    W_min=0;
end

if ~exist('W_max','var')||isempty(W_max)
    W_max=1;
end

if ~exist('mu_hat1','var')||isempty(mu_hat1)
    mu_hat1=mu_bar;
end

if ~exist('W_hat1','var')||isempty(W_hat1)
    W_hat1=zeros(size(data,1));
end

[d T]=size(data);
mu_hat=zeros(d,T);
mu_hat(:,1)=mu_hat1;
W_hat=W_hat1;
loss=zeros(1,T);

K_t=zeros(d,1);

for t=1:T
    if mod(t,5000)==0
        disp(['t=' int2str(t) ' out of ' int2str(T)]); pause(.01)
    end
    
    rho=beta/sqrt(t);
    eta=alpha/sqrt(t);
    
    %Incur loss
    loss(t)=sum(mu_hat(:,t)) - data(:,t)'*log(mu_hat(:,t));
    
    %Update W 
    W_hat_new=W_hat-rho*(ones(d,1)-data(:,t)./mu_hat(:,t))*K_t';
    W_hat_new=min(max(W_hat_new,W_min),W_max);
    K_t=(1-eta)*tau*K_t+data(:,t);
    
    %Update mu 
    if t<T
        mu_tilde=(1-eta)*mu_hat(:,t)+eta*data(:,t);
        mu_hat(:,t+1)=tau*mu_tilde+W_hat*data(:,t) + (1-tau)*mu_bar;
        mu_hat(:,t+1)=mu_hat(:,t+1)+(W_hat_new-W_hat)*K_t;
    end
    
    W_hat=W_hat_new;
end

