import numpy as np

def check_identified_anomalies(
        anomaly_locs, 
        anomalous, 
        detection_range=50, 
        min_contiguous=1
    ):
    
    # Check whether every anomaly_locs was identified in the anomalous array
    identified = np.zeros(len(anomaly_locs))
    flagged_anomalies = np.where(anomalous == 1)

    for i in range(len(anomaly_locs)):
        # Define anomaly_range as +/- anomalous_search_width of the anomaly
        anomaly_range = np.arange(int(anomaly_locs[i]) - int(detection_range), int(anomaly_locs[i]) + int(detection_range))

        if min_contiguous == 1:
            # If at least one index in the anomaly_range is identified, set identified to 1 and continue
            if np.any(np.isin(anomaly_range, flagged_anomalies)):
                identified[i] = 1
                continue

        # Find the flagged anomalies that fall within the anomaly_range
        flagged_in_range = np.intersect1d(anomaly_range, flagged_anomalies)

        # Boundary conditions
        if len(flagged_in_range) == 0: 
            continue

        # For every element in flagged_in_range, check if it belongs to a contiguous interval of length at least min_contiguous
        # in anomalous array. If yes, identified[i] = 1
        for flagged_idx in flagged_in_range: 
            contiguous_count = 1
            start_idx = flagged_idx
            end_idx = flagged_idx

            # Check if there are contiguous flagged anomalies to the left and right of the flagged_idx
            while np.isin((start_idx - 1), flagged_anomalies) and contiguous_count < min_contiguous:
                start_idx -= 1
                contiguous_count += 1
            
            while np.isin((end_idx + 1), flagged_anomalies) and contiguous_count < min_contiguous:
                end_idx += 1
                contiguous_count += 1

            # If we have a contiguous segment of at least min_contiguous, mark it as identified
            if contiguous_count >= min_contiguous:
                identified[i] = 1
                break  # No need to check further once a valid interval is found

    identified_ratio = np.sum(identified) / len(anomaly_locs)

    return identified, identified_ratio