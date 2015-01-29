function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
hAct{1} = data;
for layer = 1 : numHidden
    z{layer} = stack{layer}.W * hAct{layer} + repmat(stack{layer}.b, 1, numData);
    hAct{layer + 1} = sigmoid(z{layer});
end
layer = layer + 1;

z{layer} = stack{layer}.W * hAct{layer} + repmat(stack{layer}.b, 1, numData);
expZ = exp(z{layer});
hAct{layer + 1} = expZ ./ repmat(sum(expZ), size(expZ,1), 1); 

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
extY = zeros(ei.output_dim, numData);
extY(sub2ind(size(extY), labels', 1:size(extY,2))) = 1;
cost = - sum(sum(extY .* log(hAct{layer + 1}))) / numData;
%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
gradZ{layer} = hAct{layer + 1} - extY;
for layer = numHidden :-1: 1
    gradStack{layer + 1}.W = gradZ{layer + 1} * hAct{layer + 1}';
    gradStack{layer + 1}.W = gradStack{layer +1}.W / numData;
    gradStack{layer + 1}.b = sum(gradZ{layer + 1},2) / numData;
    gradAct{layer+1} = stack{layer + 1}.W' * gradZ{layer + 1}; 
    gradZ{layer} = gradAct{layer+1} .* hAct{layer+1} .* (1 - hAct{layer+1});
end

gradStack{layer}.W = gradZ{layer} * hAct{layer}';
gradStack{layer}.W = gradStack{layer}.W / numData;
gradStack{layer}.b = sum(gradZ{layer},2) / numData;
%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%
for layer = 1 : numHidden + 1
    gradStack{layer}.W = gradStack{layer}.W + ei.lambda * stack{layer}.W;
    cost = cost + ei.lambda/2*sum(sum(stack{layer}.W .^ 2));
end
%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



