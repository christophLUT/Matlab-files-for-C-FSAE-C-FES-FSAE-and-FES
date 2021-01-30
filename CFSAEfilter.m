function [HNA,varimp] = CFSAEfilter(data, class, entropy, l, pl)
% Implements the "Class-wise Fuzzy Similarity and Entropy" (C-FSAE) filter
% for supervised feature selection. The user selects how many of the 
% highest ranked features to retain. 
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
stdclass=zeros(length(uniclass),size(data,2)); % for standard deviation of each class and feature sigma_i,d
for i=1:length(uniclass) % class
    S2(class==uniclass(i),:)=(1-abs(bsxfun(@minus,(data(class==uniclass(i),:).^p)',(genmeanTrain(:,i).^p))')).^(1/p);
    
    % Standard deviations
    stdclass(i,:)=std(data(class==uniclass(i),:));
end


%% Adjustment of zero standard deviations to minimum positive value (avoid NaN for later divisions)
stdclass(stdclass==0)=min(stdclass(stdclass>0))/2; % set standard deviation to half of the minimum observed (excl. zeros)

% Ensure to avoid too small standard deviations
stdclass(stdclass<0.00001) = 0.00001; 

%% Entropy
H2=zeros(size(data,2),length(uniclass));
distmat=zeros(size(data,2),length(uniclass));
if entropy==1 %'DeLuca'
    for i=1:length(uniclass); % class
        
        % Calculate the Scaling Factors for a class i, denoted SF_i,d
        distmat(:,i)= (sum(((1-abs(bsxfun(@minus,genmeanTrain(:,[1:length(uniclass)]~=i),genmeanTrain(:,i))'))./(stdclass([1:length(uniclass)]~=i,:)+stdclass(i,:))).^l,1).^(1/l))/(length(uniclass)-1);
        
        % Calculate the scaled entropy value as H_i,d * SF_i,d
        H2(:,i)=entropyDeLucaMultCol(S2(class==uniclass(i),:))'.*distmat(:,i);
    end
    HNA=sum(H2,2)';
elseif entropy==2 %"Parkash"
    for i=1:length(uniclass) % class
        
        % Calculate the Scaling Factors for a class i, denoted SF_i,d
        distmat(:,i)= (sum(((1-abs(bsxfun(@minus,genmeanTrain(:,[1:length(uniclass)]~=i),genmeanTrain(:,i))'))./(stdclass([1:length(uniclass)]~=i,:)+stdclass(i,:))).^l,1).^(1/l))/(length(uniclass)-1);
        % Calculate the scaled entropy value as H_i,d * SF_i,d
        
        H2(:,i)=entropyParkashMultCol(S2(class==uniclass(i),:),2)'.*distmat(:,i);
    end
    HNA=sum(H2,2)';
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