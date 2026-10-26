function [B] = matrixpad3D(A,x)
% pad the matrix A with boundary x 
[m,n,l]=size(A);
B=ones(m+2,n+2,l+2)*x;
B(2:m+1,2:n+1,2:l+1)=A;
end

