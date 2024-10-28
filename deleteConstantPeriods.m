function selected_data = deleteConstantPeriods(selected_data, consecutive_threshold)
    % Deletes rows in `selected_data` where consecutive identical values 
    % exceed the specified threshold for any variable.
    %
    % Parameters:
    %   selected_data       - Table containing the dataset.
    %   consecutive_threshold - Integer specifying the threshold for consecutive identical values.
    %
    % Returns:
    %   selected_data       - Table with rows deleted where any variable has constant periods
    %                         exceeding the threshold.

    % Initialize a logical array to mark rows for deletion
    delete_rows = false(height(selected_data), 1);

    % Loop through each variable column (excluding date and target columns)
    for var_idx = 3:width(selected_data)

        variable_data = selected_data{:, var_idx}; % Extract data for the current variable

        start_idx = 1;
        while start_idx <= length(variable_data)
            end_idx = start_idx;

            % Find consecutive identical values
            while end_idx < length(variable_data) && variable_data(end_idx) == variable_data(end_idx + 1)
                end_idx = end_idx + 1;
            end

            % Check if the length of the constant period exceeds the threshold
            consecutive_hours = end_idx - start_idx + 1;
            if consecutive_hours > consecutive_threshold
                % Mark rows for deletion within this constant period
                delete_rows(start_idx:end_idx) = true;
            end

            % Move to the next segment
            start_idx = end_idx + 1;
        end
    end

    % Delete rows marked for deletion
    selected_data = selected_data(~delete_rows, :);

    % Display the number of deleted rows
    fprintf('Deleted %d rows with constant periods over the threshold.\n', sum(delete_rows));
end
