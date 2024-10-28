function selected_data = HighestCorrelationVariables(lagged_data, x_cols, y_col, num_lags)
    % Filter lagged variables based on highest correlation with the target variable.
    %
    % Parameters:
    %   lagged_data      - Table containing the lagged dataset.
    %   y_col            - Column index of the target variable.
    %   x_cols           - Vector specifying the column range for lagged predictors.
    %   lags_per_variable - Number of lags per variable.
    %
    % Returns:
    %   selected_data - Table containing only the date column, the target variable, and the most correlated lags.

    % extract variable names
    y_var = lagged_data.Properties.VariableNames{y_col};
    x_vars = lagged_data.Properties.VariableNames(x_cols(1): x_cols(2));

    num_variables = length(x_vars) / num_lags;

    % numeric data for correlation calculation
    x_numeric_data = lagged_data{:, x_vars};
    y_numeric_data = lagged_data{:, y_var};

    % correlation between Y and each lagged variable
    corr_matrix = corr(y_numeric_data, x_numeric_data, 'Rows', 'complete');

    % initialize selected_data
    selected_data = lagged_data(:, 1); % Assuming date is the first column
    selected_data.(y_var) = lagged_data.(y_var);

    % loop over each variable group and select the lag with highest correlation
    for var_idx = 1:num_variables
        % start and end indices for the current variable's lags
        group_start_idx = (var_idx - 1) * num_lags + 1;
        group_end_idx = group_start_idx + num_lags - 1;

        % find the lag with the highest correlation for the current group
        [~, max_corr_idx] = max(abs(corr_matrix(group_start_idx:group_end_idx)));

        % index of the selected lag in the original dataset
        selected_column_idx = x_cols(1) + (group_start_idx - 1) + (max_corr_idx - 1);
        

        % add the selected lag to the filtered data
        selected_data = [selected_data, lagged_data(:, selected_column_idx)];
    end

    disp('Filtered dataset with highest correlated lags:');
    disp(head(selected_data));
end
