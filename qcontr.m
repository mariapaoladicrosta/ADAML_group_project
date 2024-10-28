function Qcontr = qcontr(data, loadings, comp)
    % qcontr calculates the contribution of each variable to SPEx (Q)
    % Inputs:
    %   - data: Matrix of observations for specific outliers
    %   - loadings: PCA loadings from the full dataset
    %   - comp: Number of principal components to include in the calculation
    % Output:
    %   - Qcontr: Sum of squared residuals (SPEx contributions) for each variable
    
    % Calculate scores for the selected components
    score = data * loadings(:, 1:comp);
    
    % Reconstruct the data using the selected components
    reconstructed = score * loadings(:, 1:comp)';
    
    % Calculate residuals
    residuals = bsxfun(@minus, data, reconstructed);
    
    % Sum of squared residuals per variable (SPEx contributions)
    Qcontr = sum(residuals .^ 2, 1);
end
