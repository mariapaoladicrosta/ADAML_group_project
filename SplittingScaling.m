function [Calibration, Test, X_Cal, y_Cal, X_Cal_scaled, y_Cal_centered, X_test_scaled, y_test_centered] = SplittingScaling(selected_data, holdout_ratio)
    % Splits the dataset into calibration and testing sets, then standardizes and centers the data.
    %
    % Parameters:
    %   selected_data - Table containing the dataset to be split and standardized.
    %   holdout_ratio - Proportion of the data to be used as the test set (e.g., 0.2 for 20% testing data).
    %
    % Returns:
    %   X_Cal_scaled    - Standardized X matrix for the calibration set.
    %   y_Cal_centered  - Centered y vector for the calibration set.
    %   X_test_scaled   - Standardized X matrix for the test set, using calibration statistics.
    %   y_test_centered - Centered y vector for the test set, using calibration mean.
    
    % partition the data into calibration and test sets
    num_obs = height(selected_data);
    partition = tspartition(num_obs, "Holdout", holdout_ratio);
    
    Calibration = selected_data(training(partition), :);
    Test = selected_data(test(partition), :);
    
    % separate features (X) and target variable (y) for calibration and test sets
    X_Cal = table2array(Calibration(:, 3:end));
    y_Cal = table2array(Calibration(:, 2));
    
    X_test = table2array(Test(:, 3:end));
    y_test = table2array(Test(:, 2));
    
    % standardize X calibration data
    X_Cal_mean = mean(X_Cal);
    X_Cal_std = std(X_Cal);
    X_Cal_scaled = (X_Cal - X_Cal_mean) ./ X_Cal_std;
    
    % center y calibration data
    y_Cal_mean = mean(y_Cal);
    y_Cal_centered = y_Cal - y_Cal_mean;
    
    % standardize X test data using calibration mean and std
    X_test_scaled = (X_test - X_Cal_mean) ./ X_Cal_std;
    
    % center y test data using calibration mean
    y_test_centered = y_test - y_Cal_mean;
end
