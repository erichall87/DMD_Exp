% This script generates poisson data with a time evolving rate function and
% uses the DMD with additive parametric dynamics model to find both the
% time varying rates and the network W.
% Written by Eric C. Hall - 12 Sep 2014

% Generate W matrix with low rank, block structure
d=100;
for t=1:10;
    u=rand(10,1)+.1;
    W((t-1)*10+1:t*10,(t-1)*10+1:t*10)=u*u';
end
W=W.*(W>0);
W=W./max(svd(W))*.25;

%Define parameters of the dynamical model
mu_bar=.1*ones(d,1);
mu_true=mu_bar;
tau=.5;

%Generate data
T=50000;
data=zeros(d,T);
disp('Generating Data')
for t=1:T
    data(:,t)=poissrnd(mu_true);
    mu_true=tau*mu_true+W*data(:,t)+(1-tau)*mu_bar;
end
disp('done.')
% Run algorithms
disp('Running DMD exp Algorithm')
[mu_hat_DMD_exp W_hat_DMD_exp loss_DMD_exp]=DMD_Self_Excite(data,mu_bar,tau,.9,.005,0,1,ones(d,1),zeros(d));
%DMD with exp familiy is equivalent to MD when the dynamics are identity
%(i.e. tau=1 and W_hat=0 for all t)
disp('Running MD Algorithm')  
[mu_hat_MD W_hat_MD loss_MD]=DMD_Self_Excite(data,mu_bar,1,.9,0,0,1,ones(d,1),zeros(d));
disp('Running DMD Algorithm, with known W')
%DMD with exp family is equivalent to DMD with a known W when the initial
%estimate of W is W and the step size is 0 for all t
[mu_hat_DMD W_hat_DMD loss_DMD]=DMD_Self_Excite(data,mu_bar,tau,.9,0,0,1,ones(d,1),W);


%Moving average loss of algorithms
zeta=1000;
figure(1)
plot(zeta:T,[conv(loss_DMD_exp,ones(1,zeta)/zeta,'valid');...
    conv(loss_MD,ones(1,zeta)/zeta,'valid');...
    conv(loss_DMD,ones(1,zeta)/zeta,'valid')])
xlabel('Time')
ylabel('Loss')
title('Moving Avg Loss')
legend('DMD with Additive dynamics','MD','DMD with known W',0)

%Estimate of W matrix
figure(2)
subplot(121), imagesc(W,[0 max(W(:))]); colormap gray; axis off; axis square
title('True W')
subplot(122), imagesc(W_hat_DMD,[0 max(W(:))]); colormap gray; axis off; axis square
title('Estimated W')