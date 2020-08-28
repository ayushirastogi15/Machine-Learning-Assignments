function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features  -5x3
%        Theta - num_users  x num_features matrix of user features -4x3 
%        Y - num_movies x num_users matrix of user ratings of movies -5x4
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the  -5x4
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

J = sum(sum((((X * Theta') - Y).^2).*R))/2;
regterm = (lambda/2)*sum(sum(Theta.^2)) + (lambda/2)*sum(sum(X.^2));
J = J + regterm;

for i = 1:num_movies
    idx = find(R(i,:) == 1);
    %Ytemp = Y(i,:).*R(i,:); 
    Ytemp = Y(i,idx);
    Theta_temp = Theta(idx,:);
    X_grad(i,:) = ((X(i,:) * Theta_temp') - Ytemp)*Theta_temp;
    grad_reg = lambda*X(i,:);
    X_grad(i,:) = X_grad(i,:) + grad_reg;
end

for j = 1:num_users
    idx = find(R(:,j) == 1);   %5x1
    %Ytemp = Y(:,j).*R(:,j);
    Ytemp = Y(idx,j); 
    X_temp = X(idx,:);
    %Theta_grad(:,j) = X_temp'*((X * Theta')(:,j) - Ytemp);
    Theta_grad(j,:) = X_temp'*((X_temp * Theta(j,:)') - Ytemp);
    grad_reg = lambda*Theta(j,:);
    Theta_grad(j,:) = Theta_grad(j,:) + grad_reg;
end

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
