%% Your job is to implement the RICA cost and gradient

% Have a view at github cat_project https://github.com/edouardoyallon/cat_project
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%

% Forward Prop
h = W*x;
r = W'*h;

% Sparsity Cost
K = sqrt(params.epsilon + h.^2);
sparsity_cost = params.lambda * sum(sum(K));
K = 1./K;

diff=(W'*W*x-x);
costtemp=1/2*sum(sum(diff.*diff));  %note that diff is not a vector, but a 81x10000 matrix, so diff'*diff != diff.*diff
%%THE FOLLOWING LINE MAYBE SHOULD BE:cost=params.lambda*sqrt(sum(sum(h.^2))+params.epsilon)+costtemp;
cost=params.lambda*sum(sum(sqrt(h.^2)+params.epsilon))+costtemp;


outderv = W * diff;
outderv = outderv + params.lambda * (h .* K);   %Why??
W1grad = outderv * x';
Wgrad = W1grad+W*x*diff';

%my initial Wgrad is as following(same
%ashttp://ufldl.stanford.edu/tutorial/unsupervised/ExerciseRICA/)
%Wgrad = W*2*diff*x'+2*W*x*diff';
% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);