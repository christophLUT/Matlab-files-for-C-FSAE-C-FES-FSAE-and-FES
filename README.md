# Matlab code files for C-FSAE, C-FES, FSAE and FES feature selection
Class-wise Fuzzy similarity and entropy (C-FSAE), Class-wise Fuzzy Entropy and Similarity (C-FES), Fuzzy Similarity and Entropy (FSAE) and Fuzzy Entropy and Similarity (FES) filter methods to rank features according to their relevance for supervised feature selection

Matlab Code files Class-wise fuzzy similarity and entropy (C-FSAE), Class-wise fuzzy entropy and similarity (C-FES), fuzzy similarity and entropy (FSAE) and fuzzy entropy and similarity (FES) filterm methods that rank features according to their relevance for supervised feature selection in the context of classification. Files are related to the manuscript:

Lohrmann, C., Luukka, P. (2021) "Fuzzy similarity and entropy (FSAE) feature selection revisited by using intra-class entropy and a normalized scaling factor", proceedings of the NSAIS 2019 Workshop, Finland

    "CFSAEfilter", "CFESfilter", "FSAEfilter", "FESfilter" Functions that conduct the feature ranking (filter method) for supervised feature selection. The output are the feature ranking and the scores that represent the feature relevance of each feature. These are the functions deployed by someone in order to conduct feature selection on his/her classification data set.

    "Artificial_Examples_and_Feature_Selection": A file containing three artificial example cases (linked to the above mentioned manuscript) and how the four filter methods generate their feature rankings on these examples.
    
    "entropyDeLucaMultCol", "entropyParkashMultCol" Subfunctions called within the functions for the filter methods to calculate the entropy of the similarity values. These functions are called within the before-mentioned functions and do not need to be directly deployed by the user.

    "maxminscal" Function to scale the data into the unit interval [0,1]. This scaling is needed for the calculation of the similarity values within the functions for the filter methods. A user should apply this function to scale all columns into the unit interval [0,1]. An example is contained in the code file "Artificial_Examples_and_Feature_Selection".
