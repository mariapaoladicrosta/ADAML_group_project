function selected_data = R2PLSVariables(lagged_data, x_cols, y_col, PLS_components)
    % This function selects the best variables for PLS regression to maximize R².
    % It iteratively selects variables that improve the R² of the model.
    %
    % Parameters:
    %   lagged_data    - Table containing the lagged data with predictor and target variables
    %   start_col      - Starting column index for the predictor variables
    %   end_col        - Ending column index for the predictor variables
    %   target_col     - Column index of the target variable
    %   max_components - Maximum number of PLS components to use
    %
    % Returns:
    %   selected_data  - Table containing selected variables up to maximum R²

    % Extract predictors and target
    X_lagged = lagged_data{:, x_cols(1):x_cols(2)};
    Y = lagged_data{:, y_col};

    % Standardize predictors and mean-center target
    X_scaled = (X_lagged - mean(X_lagged)) ./ std(X_lagged);
    Y_centered = Y - mean(Y);

    % Initialize parameters
    [n, m] = size(X_scaled); % Get the dimensions of X
    selected_indices = []; % Indices of selected variables
    selected_names = {}; % Names of selected variables
    R2_out = []; % List to store R² values after each selection
    unselected_indices = 1:m; % Indices of unselected variables
    var_names = lagged_data.Properties.VariableNames(x_cols(1): x_cols(2)); % Get variable names from table
    TSS = sum((Y_centered - mean(Y_centered)).^2); % Total sum of squares for Y

    % Iterative variable selection process
    for v = 1:m
        R2_values = zeros(1, length(unselected_indices)); % Preallocate R² array for speed

        % Test each unselected variable
        for j = 1:length(unselected_indices)
            temp_vars = [selected_indices, unselected_indices(j)]; % Add current variable
            X_temp = X_scaled(:, temp_vars); % Select columns corresponding to current variable set

            % Calibrate PLS model with up to min(# variables, max_components) components
            num_components = min(length(temp_vars), PLS_components);
            [~, ~, ~, ~, beta] = plsregress(X_temp, Y_centered, num_components);

            % Predict Y and calculate R²
            Y_pred = [ones(n, 1), X_temp] * beta; % Include intercept
            RSS = sum((Y_centered - Y_pred).^2); % Residual sum of squares
            R2_values(j) = 1 - (RSS / TSS); % Calculate R²
        end

        % Select the variable that gives the highest R²
        [best_R2, idx] = max(R2_values);
        R2_out = [R2_out, best_R2]; % Store the best R²
        selected_indices = [selected_indices, unselected_indices(idx)]; % Add the best variable to selected
        selected_names = [selected_names, var_names{unselected_indices(idx)}]; % Store the variable name
        unselected_indices(idx) = []; % Remove selected variable from unselected set
    end

    % Find the index of the maximum R²
    [~, max_index] = max(R2_out);

    % Retrieve the variable names corresponding to the selected variables up to max_index
    final_selected_names = selected_names(1:max_index);

    % Prepare the selected data with maximum R² variables
    X_selected_data = lagged_data(:, final_selected_names);

    % Combine with date and target variable
    selected_data = lagged_data(:, 1); % Assume first column is date
    selected_data.xSilicaConcentrate_lead1 = lagged_data.xSilicaConcentrate_lead1;
    selected_data = [selected_data, X_selected_data];

    % Display final selected variable names and R² values
    % disp('Selected Variables for PLS Model with Maximum R²:');
    % disp(final_selected_names);
    % disp('R² Values for each selected variable:');
    % disp(R2_out(1:max_index));
end
