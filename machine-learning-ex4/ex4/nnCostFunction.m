function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));



% Setup some useful variables
m = size(X, 1);

% Add ones to the X data matrix
a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

y_formatted = zeros(size(y, 1), num_labels);
for c = 1:num_labels
  y_formatted(:, c) = (y == c);
end
content = -1*y_formatted'*log(a_3) - (1 - y_formatted')*log(1 - a_3);
sum = 0;
for i = 1:m
  for k = 1:num_labels
    sum = sum -1*y_formatted(i, k)*log(a_3(i, k)) - (1 - y_formatted(i, k)')*log(1 - a_3(i, k));
  end
end
reg_theta1 = 0;
for j = 1:hidden_layer_size
  for k = 1:input_layer_size
    reg_theta1 = reg_theta1 + Theta1(j,k + 1)^2;
  end
end

reg_theta2 = 0;
for j = 1:num_labels
  for k = 1:hidden_layer_size
    reg_theta2 = reg_theta2 + Theta2(j,k + 1)^2;
  end
end

J = 1/m * sum + lambda / (2*m) * (reg_theta1 + reg_theta2);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%

% Add ones to the X data matrix
a_1 = [ones(m, 1) X];
z_2 = a_1 * Theta1';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2 * Theta2';
a_3 = sigmoid(z_3);
delta_3 = zeros(m, num_labels);
delta_2 = zeros(m, hidden_layer_size + 1);
%Theta1:25x401
%Theta2:10x26

sum_delta_1 = 0;
sum_delta_2 = 0;
for i=1:m
  a_1 = X(i,:);
  a_1 = [1 a_1];%1x401
  z_2 = a_1 * Theta1';%1X25
  a_2 = sigmoid(z_2);
  a_2 = [1 a_2];%1X26
  z_3 = a_2 * Theta2';%1X10
  a_3 = sigmoid(z_3);%1X10

  delta_3 = a_3 - y_formatted(i,:);%1X10
  delta_2 = delta_3*Theta2 .* sigmoidGradient([1 z_2]);
  delta_2 = delta_2(2:end);%1X25
  sum_delta_1 = sum_delta_1 + delta_2' * a_1;
  sum_delta_2 = sum_delta_2 + delta_3' * a_2;

end
reg_1 = Theta1;
reg_1(:,1) = zeros(size(Theta1,1):1);
reg_2 = Theta2;
reg_2(:,1) = zeros(size(Theta2,1):1);
Theta1_grad = 1/m*sum_delta_1 + lambda/m*reg_1;
Theta2_grad = 1/m*sum_delta_2+ lambda/m*reg_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
