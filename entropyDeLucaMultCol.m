function c=entropyDeLucaMultCol(x)
% Calculates De Luca and Termini entropy (for multiple columns)
c=zeros(1,size(x,2));% Preprocessing
for i=1:size(x,2)
x2=x(x(:,i)<1 & x(:,i)>0,i);
c(1,i)=-sum(x2.*log(x2)+(1-x2).*log(1-x2));
end