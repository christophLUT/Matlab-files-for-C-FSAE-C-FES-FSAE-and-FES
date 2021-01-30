function c=entropyParkashMultCol(x,d,w)
% Calculates Parkash et al. entropy (for multiple columns)
c = zeros(1,size(x,2));% Preprocessing
for i=1:size(x,2)
    x2=x(x(:,i)<1,i);
    if nargin==2
        w=ones(size(x2,1),1);
    end
    if d==1
        c(1,i)=sum(w.*gsubtract(sin(pi*x2./2)+sin(pi*(1-x2)./2),ones(size(x2,1),1)));
    elseif d==2
        c(1,i)=sum(w.*gsubtract(cos(pi*x2./2)+cos(pi*(1-x2)./2),ones(size(x2,1),1)));
    end
end

end