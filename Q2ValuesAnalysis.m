function [model, avg_positive_Q2] = Q2ValuesAnalysis(Calibration, model, batch_sizes, max_components)
    % Analyzes Q² values across different batch sizes and latent variables.
    %
    % Parameters:
    %   Calibration    - Original calibration dataset with date in the first column.
    %   model          - Model structure to store results for each batch size and latent variable.
    %   batch_sizes    - Array of batch sizes (number of samples in each batch).
    %   max_components - Maximum number of latent variables (components) to evaluate.
    %
    % Returns:
    %   model          - Updated model structure with negative Q² values, indices, and dates.
    %   avg_positive_Q2 - Matrix of average positive Q² values for each batch size and latent variable.

    % Initialize the matrix for average positive Q² values
    avg_positive_Q2 = zeros(length(batch_sizes), max_components);

    % Loop over each batch size and each number of latent variables
    for i = 1:length(batch_sizes)
        m_value = batch_sizes(i); % Current window size for this iteration
        for j = 1:max_components
            % Extract Q² values for the current batch size and component count
            Q2_values = model(m_value).ncomp(j).Q2;
            
            % Identify positive Q² values and calculate their average
            positive_Q2_values = Q2_values(Q2_values > 0);
            avg_positive_Q2(i, j) = mean(positive_Q2_values);
            
            % Identify and store negative Q² values and their indexes
            negative_Q2_indices = find(Q2_values <= 0);
            negative_Q2_values = Q2_values(negative_Q2_indices);
            
            % Adjust segment indices to match the original calibration dataset indices
            original_indices = negative_Q2_indices + m_value + 1;
            
            % Retrieve dates from the Calibration dataset using adjusted indices
            dates_for_indices = Calibration{original_indices, 1};

            % Save negative Q² indices, values, and dates in model structure
            model(m_value).ncomp(j).negative_Q2 = table(negative_Q2_values', negative_Q2_indices', original_indices', dates_for_indices, ...
                'VariableNames', {'Q2_value', 'segmentIndex', 'originalIndex', 'Dates'});
        end
    end
end
