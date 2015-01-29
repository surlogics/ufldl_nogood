function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
  
  
  %%X:n x m; theta: (numclass-1) x n;
esub=exp(theta'*X); %esub: (numclass-1) x m
esub=[esub;ones(1,m)];  %add one row and get: numclass x m
posterior=bsxfun(@rdivide,esub,sum(esub,1));    %posterior: numclass x m
logterm=log(posterior); %logterm: numclass x m
I=sub2ind(size(logterm),y,1:size(logterm,2));   %for every column in logterm, choose an element based on the value of y(i)
values = logterm(I);
f = -sum(values);

indxm=zeros(num_classes,m);
% indx=sub2ind(size(indxm),1:size(indxm),y);
indxm(I)=1;
matrx=X*(indxm-posterior)'; %sum over i in the gradient forumula at: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
g=-matrx(:,1:end-1);



  g=g(:); % make gradient a vector for minFunc

