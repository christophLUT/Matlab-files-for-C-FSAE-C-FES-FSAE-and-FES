function [HNA, varimp] = CFESfilter(data, class, entropy, pl)
% Implements the "Class-wise Fuzzy Entropy and Similarity" (C-FES) filter 
% for supervised feature selection. The user selects how many of the 
% highest ranked features to retain. 
%
% Inputs:
%   data: All data with observations as rows and features (= variables) as
%         columns (no class labels included here)[matrix]
%   class: class labels for data [vector]
%   entropy: 1 for DeLuca & Termini entropy, 2 for Parkash et al entropy
%   pl: plotting - 1 for plotting the variable importance, 0 for no plot
%
% Outputs:
%   HNA: scaled entropy values (relatively high values indicate less / not important
%   features)
%   varimp: variable importance based on the scaled entropy - within the interval [0,1]
%   (from 1 - most important, to 0 - least important)


% Please cite: 
% Lohrmann, C. & Luukka, P. (2021), "Fuzzy similarity and entropy (FSAE)
% feature selection revisited by using intra-class entropy and a normalized 
% scaling factor", Proceedings of the NSAIS 2019 Workshop.


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


%% Similarity - only observations in each class to their own ideal vector
S2=zeros(size(data,1),size(data,2));
for i=1:length(uniclass) % class
    S2(class==uniclass(i),:)=(1-abs(bsxfun(@minus,(data(class==uniclass(i),:).^p)',(genmeanTrain(:,i).^p))')).^(1/p);
end


%% Entropy
HNA=zeros(1,size(data,2));
if entropy==1 %'DeLuca'
    for i=1:length(uniclass) % classes
        HNA=HNA+entropyDeLucaMultCol(S2(class==uniclass(i),:));
    end
elseif entropy==2 %"Parkash"
    for i=1:length(uniclass) % classes
        HNA=HNA+entropyParkashMultCol(S2(class==uniclass(i),:),1);
    end
else
end


%% Variable importance (based on scaled entropy values)
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
