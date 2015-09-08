function [zRand,SR,SAR,VI] = zrand(part1,part2)

if size(part1,1)==1,
    part1=part1';
end
if size(part2,1)==1,
    part2=part2';
end
if length(part1)~=length(part2),
    disp('ERROR: partitions not of equal length')
    return
end

%Generate contingency table and calculate row/column marginals
nij=sparse(part1+1,part2+1,1);
ni=sum(nij,2);
nj=sum(nij,1);

%Identify total number of elements, n, numbers of pairs, M, and numbers of
%classified-same pairs in each partition, M1 and M2.
n=length(part1);
M=n*(n-1)/2;
M1=sum(ni.^2-ni)/2;
M2=sum(nj.^2-nj)/2;

%Pair counting types:
a=full(sum(sum(nij.^2-nij)))/2; %same in both
b=M1-a;                         %same in 1, diff in 2
c=M2-a;                         %same in 2, diff in 1
d=M-(a+b+c);                    %diff in both

%Rand and Adjusted Rand indices:
SR=(a+d)/(a+b+c+d);
meana=M1*M2/M;
SAR=(a-meana)/((M1+M2)/2-meana);

%Calculate variance of a and z-score of Rand:
%C2=sum(nj.^3);
%C2=sum(nj.^3);
%vara = (C1*C2*(n+1) - C1*(4*M2^2+(6*n+2)*M2+n^2+n) - C2*(4*M1^2+(6*n+2)*M1+n^2+n))...
%    /(n*(n-1)*(n-2)*(n-3)) + M/16 - (4*M1-2*M)^2*(4*M2-2*M)^2/(256*M^2) +...
%    (8*(n+1)*M1-n*(n^2-3*n-2))*(8*(n+1)*M2-n*(n^2-3*n-2))/(16*n*(n-1)*(n-2)) +...
%    (16*M1^2-(8*n^2-40*n-32)*M1+n*(n^3-6*n^2+11*n+10))*...
%    (16*M2^2-(8*n^2-40*n-32)*M2+n*(n^3-6*n^2+11*n+10))/(64*n*(n-1)*(n-2)*(n-3));
C1=4*sum(ni.^3)-8*(n+1)*M1+n*(n^2-3*n-2);
C2=4*sum(nj.^3)-8*(n+1)*M2+n*(n^2-3*n-2);
vara = M/16 - (4*M1-2*M)^2*(4*M2-2*M)^2/(256*M^2) + C1*C2/(16*n*(n-1)*(n-2)) + ...
    ((4*M1-2*M)^2-4*C1-4*M)*((4*M2-2*M)^2-4*C2-4*M)/(64*n*(n-1)*(n-2)*(n-3));
zRand=(a-meana)/sqrt(vara);

%Variation of Information:
c1=unique(part1);
c2=unique(part2);
H1=0; H2=0; I=0;
for i=c1';
    pi=ni(i+1)/n;
    H1=H1-pi*log(pi);
    for j=c2';
        if nij(i+1,j+1),
            pj=nj(j+1)/n;
            pij=nij(i+1,j+1)/n;
            I=I+pij*log(pij/pi/pj);
        end
    end
end
for j=c2';
    pj=nj(j+1)/n;
    H2=H2-pj*log(pj);
end
VI = (H1+H2-2*I);
