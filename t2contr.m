function T2varcontr = t2contr(data, loadings, latent, comp)
    % t2contr calculates the contribution of each variable to T²
    % Inputs:
    %   - data: Matrix of observations for specific outliers
    %   - loadings: PCA loadings from the full dataset
    %   - latent: Eigenvalues from PCA, representing variance explained by each component
    %   - comp: Number of principal components to include in the calculation
    % Output:
    %   - T2varcontr: Sum of absolute T² contributions for each variable
    
    % Calculate scores for the selected components
    score = data * loadings(:, 1:comp);
    
    % Standardize scores based on the eigenvalues (latent)
    standscores = bsxfun(@times, score(:, 1:comp), 1 ./ sqrt(latent(1:comp, :))');
    
    % Calculate T² contributions for each variable
    T2contr = abs(standscores * loadings(:, 1:comp)');
    
    % Sum contributions per variable
    T2varcontr = sum(T2contr, 1);
end
