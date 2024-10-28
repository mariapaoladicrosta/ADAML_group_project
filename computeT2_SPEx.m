function [T2_reduced, SPEx, outliers_T2, outliers_SPEx, control_limits_T2,control_limits_SPEx] = computeT2_SPEx(X, explained_variance_threshold)
    % Computes the T² and SPEx statistics based on PCA and identifies outliers for each.
    %
    % Parameters:
    %   X                          - Input data matrix.
    %   explained_variance_threshold - Threshold for cumulative variance to select components.
    %
    % Returns:
    %   T2_reduced                 - T² statistic values for each sample.
    %   SPEx                       - SPEx statistic values for each sample.
    %   outliers_T2                - Indices of T² outliers (exceeding 3 std control limit).
    %   T2_values                  - Values of T² for the outliers.
    %   outliers_SPEx              - Indices of SPEx outliers (exceeding 3 std control limit).
    %   SPEx_values                - Values of SPEx for the outliers.

    % Step 1: Standardize the data
    X_standardized = (X - mean(X)) ./ std(X);

    % Step 2: Perform PCA on standardized data
    [coeff, score, latent, ~, explained] = pca(X_standardized);

    % Step 3: Determine number of components based on cumulative explained variance
    num_components = find((cumsum(explained)) / sum(explained) >= explained_variance_threshold, 1);

    % Step 4: Calculate the reduced T² statistic
    T2_reduced = sum((score(:, 1:num_components) ./ sqrt(latent(1:num_components)')) .^ 2, 2);

    % Step 5: Calculate the SPEx statistic
    reconstructed_X = score(:, 1:num_components) * coeff(:, 1:num_components)'; % Reconstruction
    residuals = X - reconstructed_X; % Residuals after reconstruction
    SPEx = sum(residuals .^ 2, 2); % SPEx is the sum of squared residuals for each sample

    % Step 6: Define control limits and identify outliers
    mean_T2 = mean(T2_reduced);
    std_T2 = std(T2_reduced);
    control_limit_T2_2std = mean_T2 + 2 * std_T2;
    control_limit_T2_3std = mean_T2 + 3 * std_T2;
    control_limits_T2 = [control_limit_T2_2std,control_limit_T2_3std];
    outliers_T2 = find(T2_reduced > control_limit_T2_3std);

    mean_SPEx = mean(SPEx);
    std_SPEx = std(SPEx);
    control_limit_SPEx_2std = mean_SPEx + 2 * std_SPEx;
    control_limit_SPEx_3std = mean_SPEx + 3 * std_SPEx;
    control_limits_SPEx = [control_limit_SPEx_2std,control_limit_SPEx_3std];
    outliers_SPEx = find(SPEx > control_limit_SPEx_3std);
end
