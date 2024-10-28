function VIP = computeVIP(XLoadings, yScore, numComponents, TSS)
    % Compute the Variable Importance in Projection (VIP) scores for each predictor
    %
    % Inputs:
    %   - XLoadings: Loadings of the predictors from PLS regression
    %   - yScore: Score matrix for the response variable
    %   - numComponents: Number of components to consider
    %   - TSS: Total sum of squares of the response variable (y)
    %
    % Output:
    %   - VIP: Vector of VIP scores for each predictor

    % Number of predictors
    numPredictors = size(XLoadings, 1);
    
    % Calculate the sum of squares for each component in yScore
    SSY = sum(yScore(:, 1:numComponents).^2, 1);
    
    % Initialize the VIP vector
    VIP = zeros(numPredictors, 1);

    % Loop through each predictor to calculate its VIP score
    for j = 1:numPredictors
        VIP(j) = sqrt(numPredictors * sum((XLoadings(j, 1:numComponents).^2 .* SSY) / TSS));
    end
end
