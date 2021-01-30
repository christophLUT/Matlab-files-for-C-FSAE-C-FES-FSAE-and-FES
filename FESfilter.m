function [HNA, varimp] = FESfilter(data, class, entropy, pl)
% Implements the "Fuzzy Entropy and Similarity" (FES) filter for 
% supervised feature selection, in some works referred to as "Feature 
% Selection by Luukka (2011)". The user selects how many of the highest 
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
% Luukka, P. (2011), "Feature selection using fuzzy entropy measures with 
% similarity classifier", Expert Systems with Applications, Vol. 38, 
% pp. 4600–4607.


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
S2=zeros(size(data,1),size(data,2),length(unique(class)));
for j=1:size(data,1); % samples
    for i=1:size(data,2); % features
        for k=1:length(uniclass); % class
            S2(j,i,k)=(1-abs((genmeanTrain(i,k)^p)-((data(j,i))^p)))^(1/p);
        end
    end
end

% If not between 0 and 1 since differently scaled input
S2Norm=[];
for g=1:size(S2,3)
S2Norm=[S2Norm;S2(:,:,g)]; % not not normalized again
end


%% Entropy
H=zeros(size(data,2),1);
if entropy==1 %'DeLuca'
    for i=1:size(data,2); % features
        H(i)=entropyDeLucaMultCol(S2Norm(:,i));
    end
elseif entropy==2 %"Parkash"
    for i=1:size(data,2); % features
        H(i)=entropyParkashMultCol(S2Norm(:,i),1);
    end
else
end
    
HNA=zeros(1,size(data,2));
for i=1:size(data,2); % features
HNA(1,i)=sum(H(i,:)); %/max(dataTrain(:,1)); % scale by amount of classes
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
