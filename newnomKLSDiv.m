function [D1] = newnomKLSDiv(t,q)
st=sum(t,1);
sq=sum(q,1);
t=t./repmat(st,size(t,1),1);
q=q./repmat(sq,size(q,1),1);

D=zeros(size(t,2),size(q,2));
for i=1:size(t,2)
    for j=1:size(q,2)
        for k=1:size(q,1)
            D(i,j)=D(i,j)+log(q(k,j)/t(k,i))*q(k,j);
        end
        for k=1:size(t,1)
            D(i,j)=D(i,j)+log(t(k,i)/q(k,j))*t(k,i);
        end
    end
end

for i=1: size(D,1)
    dmin=min(D(i,:));
    dmax=max(D(i,:));
    for j=1: size(D,2)
        D1(i,j)=(D(i,j)-dmin)/(dmax-dmin);
    end
end

for j=1:size(D,2)
    dmin=min(D(:,j));
    dmax=max(D(:,j));
    for i=1: size(D,1)
        D1(i,j)=(D(i,j)-dmin)/(dmax-dmin);
    end
end


