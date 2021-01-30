function [HNA,varimp] = FSAEfilter(data, class, entropy, l, pl)
% Implements the "Fuzzy Similarity and Entropy" (FSAE) filter for 
% supervised feature selection. The user selects how many of the highest 
% ranked features to retain. 
%
% Inputs:
%   data: All data with observations as rows and features (= variables) as
%         columns (no class labels included here)[matrix]
%   class: class labels for data [vector]
%   entropy: 1 for DeLuca & Termini entropy, 2 for Parkash et al entropy
%   l: parameter for the scaling factor
%   pl: plotting - 1 for plotting the variable importance, 0 for no plot
%
% Outputs:
%   HNA: scaled entropy values (relatively high values indicate less / not important
%   features)
%   varimp: variable importance based on the scaled entropy - within the interval [0,1]
%   (from 1 - most important, to 0 - least important)


% Please cite: 
% Lohrmann, C., Luukka, P., Jablonska-Sabuka, M., Kauranne, T. (2018), 
% "A combination of fuzzy similarity measures and fuzzy entropy measures
% for supervised feature selection", Expert Systems with Applications, Vol.
% 10, pp.216-236.


%% Parameters for feature selection (can be adjusted by the user)
m = 1; % generalized mean parameter
p = 1; % lukasiewicz parameter


%% Scaling into unit interval [0,1]
novariance = var(data)==0; % indices of features with no variance [in case not excluded from the data]
data(:,~novariance) = maxminscal(data(:,~novariance)); % conduct the actual scaling
data(:,novariance) = 0; % simply replace values by zero


%% Ideal Vectors
genmeanTrain=zeros(size(data,2),length(unique(class))); % pre-processing
uniclass=unique(class); % vector of unique classes
for i=1:length(uniclass) % all classes 
    genmeanTrain(:,i)=(mean(data(class==uniclass(i),:).^m,1)).^(1/m);
    % Generalized mean for each feature (row) for each class (column) - rows = observations; columns=class
end


%% Similarity - all observations with all ideal vectors
S2=zeros(size(data,1),size(data,2),length(uniclass));
for j=1:size(data,1) % samples
    for i=1:size(data,2) % features
        for k=1:length(uniclass) % class
            S2(j,i,k)=(1-abs((genmeanTrain(i,k)^p)-((data(j,i))^p)))^(1/p);
        end
    end
end


%% Entropy
H2=zeros(size(data,2),length(uniclass));
distmat=zeros(size(data,2),length(uniclass));
if entropy==1 %'DeLuca'
    for k=1:length(uniclass) % class
        for i=1:size(data,2) % features
            distmat(i,k)=1-((sum(abs(gsubtract(repmat(genmeanTrain(i,k),size(genmeanTrain(i,:),2)-1,1),genmeanTrain(i,[1:length(unique(class))]~=k)')).^l).^(1/l))/(length(unique(class))-1)); % abs value of distances to the gmean val of that feature for class k
            H2(i,k)=entropyDeLucaMultCol(S2(:,i,k))*distmat(i,k); 
        end
    end
elseif entropy==2 %"Parkash"
    for k=1:length(uniqueclass) % class
    for i=1:size(data,2) % features
        distmat(i,k)=1-((sum(abs(gsubtract(repmat(genmeanTrain(i,k),size(genmeanTrain(i,:),2)-1,1),genmeanTrain(i,[1:length(unique(class))]~=k)')).^l).^(1/l))/(length(unique(class))-1)); % abs value of distances to the gmean val of that feature for class k
        H2(i,k)=entropyParkashMultCol(S2(:,i,k),2)*distmat(i,k);
    end
    end
else
end

% Summation of entropies for each feature (over the classes)
HNA=zeros(1,size(data,2));
for i=1:size(data,2) % features
HNA(1,i)=sum(H2(i,:)); %/max(dataTrain(:,1)); % can be scaled by amount of classes (does not affect feature ranking)
end


%% Calculation of feature importance
varimp=1-HNA/sum(HNA);
varimp=maxminscal(varimp);


%% Visualization
if pl==1
    figure
    bar(varimp,'FaceColor',[0.75 0.75 0.75]), hold on
    axis([0 size(data,2)+1 0 1.1])
    xlabel('Features')
    ylabel('Feature Importance')
    grid on
    title('Variable Importance','Fontweight','bold')
end

end




% Christoph Lohrmann, Lappeenranta University of Technology
% Project together with Prof. Pasi Luukka