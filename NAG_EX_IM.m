% % % clear all;
% % % clc;
% close all;
%% real data set
L=0*2*10^(-10);
N=10000;

a=1;
%%
% load('colon_cancer_A');
% load('colon_cancer_label');
% W=A';
% H=Y;
% c_W =size(W,1);
% for i=1:c_W
% Ma= max(W(i,:));
% Mi = min(W(i,:));
%  W(i,:) = (W(i,:) - Mi)/(Ma-Mi);
%  H(i) = (H(i,:) - Mi)/(Ma-Mi);
% end
% load('X_a1a');
% load('y_a1a_label');
% % % X=rand(10,10);
% % % y=rand(10,1);
% % % for i=1:10
% % %     if y(i)>=0.5
% % %         y(i)=1;
% % %     else
% % %         y(i)=0;
% % %     end
% % % end
% % % 
% % % W=X;
% % % H=y; 
d=size(W,2);
%% initial conditions
% % % z11=rand(d,1); z22=zeros(d,1); z33=1;
% % % z0=[z11;z22;z33];
%%
SW1=1;
SW2=2;
ImF1=zeros(N+1,1);
ImF2=zeros(N+1,1);
ImF3=zeros(N+1,1);
ExFF=zeros(N+1,1);
ExF_NAG=zeros(N+1,1);
ExF_GD=zeros(N+1,1);
% 
% h1=0.02;
% h2=0.1;
% h3=0.2;
% h_NAG=0.02;
% h_GD=0.02;

h1=0.01;
h2=0.02;
h3=0.1;
h_NAG=0.01;
h_GD=0.01;
ImX1=Im_1s1o_trape(a, h1, N, z0, d, W, H, L);
ImX2=gauss_bfgs_crj(a, h2, N, z0, d, W, H, L);
% Exp_X=Ex_2s2o_Kutta(a, hh, N, z0, d, W, H, L);
ImX3=Im_3s6o_gauss(a, h3, N, z0, d, W, H, L);
X_NAG=nesterov(h_NAG, N, z0, d, W, H, L, SW2);
X_GD=nesterov(h_GD, N, z0, d, W, H, L, SW1);
for j=1:N+1
%% loss 
ImF1(j) = Fu(ImX1(:, j), W, H, L);
ImF2(j) = Fu(ImX2(:, j), W, H, L);
ImF3(j) = Fu(ImX3(:, j), W, H, L);
% ExFF(j) = Fu(Exp_X(:, j), W, H, L);
ExF_NAG(j) = Fu(X_NAG(:, j), W, H, L);
ExF_GD(j) = Fu(X_GD(:, j), W, H, L);
end
% 
figure
semilogy(1:N+1, ImF1,'r-','LineWidth', 1.5);hold on
semilogy(1:N+1, ImF2,'b-','LineWidth', 1.5);hold on
semilogy(1:N+1, ImF3,'r--','LineWidth', 1.5);hold on
% semilogy(1:N+1, ExFF,'b--','LineWidth', 1.5);
semilogy(1:N+1, ExF_NAG,'k:','LineWidth', 1.5);hold on;
semilogy(1:N+1, ExF_GD,'k-.','LineWidth', 1.5);
xlabel('Iterations', 'FontSize',16);
ylabel('Objective','FontSize',16);
%title('Minimizing regularized quadratic function on  set','FontSize',16);
legend({'Im s=1','Im s=2','Im s=3','NAG','GD'},'FontSize',16); %'Ex-2s2o-Kutta'
set(gca,'FontSize',16);