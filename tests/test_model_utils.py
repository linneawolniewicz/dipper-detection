# imports
import unittest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock
from dipper.model_utils.utils import *

class TestCheckIdentifiedAnomalies(unittest.TestCase):
    print("Testing check_identified_anomalies")

    def test_identifies_single_flagged_point(self):
        anomaly_locs = np.array([10, 50])
        anomalous = np.zeros(100)
        anomalous[9] = 1  # One point flagged near the first anomaly
        anomalous[48:52] = 1  # Several points flagged near the second anomaly
        
        identified, identified_ratio = check_identified_anomalies(anomaly_locs, anomalous, detection_range=2, min_contiguous=1)

        self.assertTrue(np.all(identified == [1, 1]), f"Expected both anomalies to be identified, got {identified}")
        self.assertEqual(identified_ratio, 1, f"Expected identified_ratio to be 1, got {identified_ratio}")

    def test_min_contiguous_larger_than_flagged(self):
        anomaly_locs = np.array([30])
        anomalous = np.zeros(100)
        anomalous[28:30] = 1  # Only 2 points flagged
        
        identified, identified_ratio = check_identified_anomalies(anomaly_locs, anomalous, detection_range=5, min_contiguous=3)

        self.assertTrue(np.all(identified == [0]), f"Expected no anomalies to be identified, got {identified}")
        self.assertEqual(identified_ratio, 0, f"Expected identified_ratio to be 0, got {identified_ratio}")

    def test_mixed_anomaly_identification(self):
        anomaly_locs = np.array([20, 60, 90])
        anomalous = np.zeros(100)
        anomalous[18:22] = 1  # Contiguous block of 4 points (should be identified)
        anomalous[59] = 1  # Single flagged point (should not be identified)
        anomalous[88:90] = 1  # 2 contiguous points (should not be identified with min_contiguous = 3)
        
        identified, identified_ratio = check_identified_anomalies(anomaly_locs, anomalous, detection_range=5, min_contiguous=3)

        self.assertTrue(np.all(identified == [1, 0, 0]), f"Expected [1, 0, 0], got {identified}")
        self.assertEqual(identified_ratio, 1/3, f"Expected identified_ratio to be 1/3, got {identified_ratio}")

    def test_detection_range_boundary(self):
        anomaly_locs = np.array([40])
        anomalous = np.zeros(100)
        anomalous[30:35] = 1  # Flagged points outside of detection range
        
        identified, identified_ratio = check_identified_anomalies(anomaly_locs, anomalous, detection_range=3, min_contiguous=1)

        self.assertTrue(np.all(identified == [0]), f"Expected no anomalies to be identified, got {identified}")
        self.assertEqual(identified_ratio, 0, f"Expected identified_ratio to be 0, got {identified_ratio}")

    def test_detection_contiguous_on_boundary(self):
        anomaly_locs = np.array([40])
        anomalous = np.zeros(100)
        anomalous[31:38] = 1  # Flagged points overlapping detection range
        
        identified, identified_ratio = check_identified_anomalies(anomaly_locs, anomalous, detection_range=3, min_contiguous=5)

        self.assertTrue(np.all(identified == [0]), f"Expected no anomalies to be identified, got {identified}")
        self.assertEqual(identified_ratio, 0, f"Expected identified_ratio to be 0, got {identified_ratio}")

    def test_no_flagged_points(self):
        anomaly_locs = np.array([25, 75])
        anomalous = np.zeros(100)  # No flagged points at all

        identified, identified_ratio = check_identified_anomalies(anomaly_locs, anomalous, detection_range=10, min_contiguous=1)

        self.assertTrue(np.all(identified == [0, 0]), f"Expected no anomalies to be identified, got {identified}")
        self.assertEqual(identified_ratio, 0, f"Expected identified_ratio to be 0, got {identified_ratio}")