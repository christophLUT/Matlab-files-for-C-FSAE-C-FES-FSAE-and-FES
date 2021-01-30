%% Artificial Examples and Feature Selection
%
% Christoph Lohrmann, PhD, Pasi Luukka, DSc (Tech)
% Manuscript: "Fuzzy similarity and entropy (FSAE) feature selection 
% revisited by using intra-class entropy and a normalized scaling factor"
% Proceedings of the NSAIS 2019 Workshop (Lappeenranta, Finland)
%
% January, 2021

clear all
close all
clc

%% Short Instruction
% 1. Please select one of the examples in "Part 1" and run the code. The
% example data will already be scaled into the [0,1] interval and a figure
% illustrating the sample points will appear.
% 2. Please run "Part 2" to conduct all the feature rankings and compare
% the results.

%% PART 1: Select one of the three articifial examples
%% Example 1: One decision region per class, 3-class problem, complete overlap for first feature

% Number of Observations per Class
num=1000;

% Generate Data from Normal Distribution using the Mean Values and Covariances
x1=[50+randn(num,1)/100 10+randn(num,1)];
x2=[50+randn(num,1)/100 50+randn(num,1)];
x3=[50+randn(num,1)/100 100+randn(num,1)];
data=[x1;x2;x3];
class=[repmat(1,num,1);repmat(2,num,1); repmat(3,num,1)];

% Scale the Data into [0,1] for each column (=variable)
data=maxminscal(data);


% Visualization
plot(data(class==1,1),data(class==1,2),'r.','MarkerSize',5), hold on
plot(data(class==2,1),data(class==2,2),'b.','MarkerSize',5),
plot(data(class==3,1),data(class==3,2),'g.','MarkerSize',5),
xlim([-0.1 1.1])
ylim([-0.1 1.1])
title('Example 1')
xlabel('Feature 1')
ylabel('Feature 2')
legend('Class 1','Class 2','Class 3','Location','southoutside','Orientation','horizontal')
grid on

%% Example 2: One decision region per class, 3-class problem, moderate overlap for the first feature

% Number of Observations per Class
num=1000;

% Generate Data from Normal Distribution using the Mean Values and Covariances
x1=[80+10*randn(num,1) 10+randn(num,1)];
x2=[100+10*randn(num,1) 20+randn(num,1)];
x3=[120+10*randn(num,1) 30+randn(num,1)];
data=[x1;x2;x3];
class=[repmat(1,num,1);repmat(2,num,1); repmat(3,num,1)];

% Scale the Data into [0,1] for each column (=variable)
data=maxminscal(data);

% Visualization
plot(data(class==1,1),data(class==1,2),'r.','MarkerSize',5), hold on
plot(data(class==2,1),data(class==2,2),'b.','MarkerSize',5),
plot(data(class==3,1),data(class==3,2),'g.','MarkerSize',5),
xlim([-0.1 1.1])
ylim([-0.1 1.1])
title('Example 2')
xlabel('Feature 1')
ylabel('Feature 2')
legend('Class 1','Class 2','Class 3','Location','southoutside','Orientation','horizontal')
grid on


%% Third example: four class problem with single decision region each, different standard deviations

% Specify Mean Values and Covariance Matrices
mu(1,:) = [5,5];
sigma(:,:,1) = [2 0; 0 0.05];
mu(2,:) = [5,10];
sigma(:,:,2) = [0.05 0; 0 2];
mu(3,:) = [5,15];
sigma(:,:,3) = sigma(:,:,1);
mu(4,:) = [5,20];
sigma(:,:,4) = sigma(:,:,2);
classno=[1,2,3,4];

% Number of Observations per Class
num=500;

% Generate Data from Normal Distribution using the Mean Values and Covariances
data=[];
class=[];
for o=1:size(classno,2);
temp = mvnrnd(mu(o,:),sigma(:,:,o),num); % SIGMA
data=[data; temp];
class=[class; repmat(classno(o),num,1)];
end

% Scale the Data into [0,1] for each column (=variable)
data=maxminscal(data);


% Visualization
plot(data(class==1,1),data(class==1,2),'r.','MarkerSize',5), hold on
plot(data(class==2,1),data(class==2,2),'b.','MarkerSize',5),
plot(data(class==3,1),data(class==3,2),'g.','MarkerSize',5),
plot(data(class==4,1),data(class==4,2),'m.','MarkerSize',5),
xlim([-0.1 1.1])
ylim([-0.1 1.1])
title('Example 3')
xlabel('Feature 1')
ylabel('Feature 2')
legend('Class 1','Class 2','Class 3','Class 4','Location','southoutside','Orientation','horizontal')
grid on


%% PART 2: Run the feature ranking for C-FSAE, FSAE, C-FES, and FES
%% Use Filter Methods for Feature Ranking (Supervised Learning)

% Select parameters
entropy = 1; % 1: De Luca Entropy, 2: Parkash et al. Entropy
pl = 0; % 0: No plot, 1: plot of the feature importance values
l_CFSAE = 2; % 'l' parameter for CFSAE (l = 2 recommended)
l_FSAE = 1; % 'l' paramter for FSAE


% Calculation of the feature rankings

% Class-wise Fuzzy similarity and entropy (C-FSAE) feature selection
[CFSAEscore,~] = CFSAEfilter(data, class, entropy, l_CFSAE, pl); 
[~, rankedCFSAE]=sort(CFSAEscore,'ascend') % Features Ranked from most relevant to least relevant / irrelevant

% Fuzzy similarity and entropy (FSAE) feature selection
[FSAEscore,~] = FSAEfilter(data, class, entropy, l_FSAE, pl);
[~, rankedFSAE]=sort(FSAEscore,'ascend') % Features Ranked from most relevant to least relevant / irrelevant

% Class-wise Fuzzy entropy and similarity (C-FES) feature selection
[CFESscore, ~] = CFESfilter(data, class, entropy, pl); % CFES
[~, rankedCFES]=sort(CFESscore,'ascend')

% Fuzzy entropy and similarity (FES) feature selection
[FESscore, ~] = FESfilter(data, class, entropy, pl); % FES
[~, rankedFES]=sort(FESscore,'ascend')
    


