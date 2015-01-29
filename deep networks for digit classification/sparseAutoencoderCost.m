function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% size(data, 1) % 64
% size(W1) % 25 64
% size(W2) % 64 25
% size(b1) % 25 1
% size(b2) % 64 1
m = size(data,2);
% 
% for i = 1:visibleSize
%     z2(:,i) = W1*reshape(data(:,i),[visibleSize,1])+b1;
%     a2(:,i) = sigmoid(z2(:,i));
%     z3(:,i) = W2*reshape(a2(:,i),[visibleSize,1])+b1;
%     a3(:,i) = sigmoid(z3(:,i));
% end
% for i = 1:visibleSize
%     delta3(i) = -(data(:,i)-a3(:,i)).*fprime(z3(i));
%     delta2(i) = (W2(i)'*delta3(i)).*fprime(z2(i));
%     for j = 1:hiddenSize
%         W2grad(i,j) = delta3(i)*a2(j)';
%         W1grad(i,j) = delta2(i)*data(:,j)';
%         b2grad = delta3(i);
%         b1grad = delta2(i);
%     end
% end

z2 = W1*data + repmat(b1,1,m);
a2 = sigmoid(z2);% 25 10000
z3 = W2*a2 + repmat(b2,1,m);
a3 = sigmoid(z3);% 64 10000
rou_hat =  sum(a2,2)/m;
delta3 = -(data - a3) .* fprime(z3);% 64 10000
J_simple = sum(sum((a3-data).^2)) / (2*m);
sparsepen = kl(sparsityParam,rou_hat);
d2_pen = kl_div(rou_hat,sparsityParam);
delta2 = ((W2' * delta3)+beta*repmat(d2_pen,1,m)) .* fprime(z2);% 25 10000
reg = sum(W1(:).^2) + sum(W2(:).^2);
cost = J_simple + beta * sparsepen + lambda * reg / 2;
nablaW1 = delta2 * data';
nablab1 = delta2;
nablaW2 = delta3 * a2';
nablab2 = delta3;
W1grad = nablaW1 ./ m + lambda .* W1;% 25 64
W2grad = nablaW2 ./ m + lambda .* W2;% 25 64
b1grad = sum(nablab1, 2) ./ m;
b2grad = sum(nablab2, 2) ./ m;
















%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end
function ans = kl(r, rh)
ans = sum(r .* log(r ./ rh) + (1-r) .* log( (1-r) ./ (1-rh)));
end
function fp = fprime(x)
    fp = sigmoid(x).*(1-sigmoid(x));
end
function div = kl_div(rh,r)
    div = (1-r)./(1-rh) - r./rh;
end