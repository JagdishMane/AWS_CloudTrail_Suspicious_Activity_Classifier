import copy
import unittest
import numpy as np
from common import normalize, weak_label, fit_encoder, transform, feature_names, TARGET

def event(**changes):
    d = {"eventTime":"2024-01-01T00:00:00Z", "eventName":"DescribeInstances", "eventSource":"ec2.amazonaws.com", "userIdentity":{"type":"IAMUser"}, "sourceIPAddress":"192.0.2.1"}
    d.update(changes)
    return d

class ContractTests(unittest.TestCase):
    def test_missing_mfa_is_unknown(self):
        r = normalize(event(eventName="AssumeRole", eventSource="sts.amazonaws.com"))
        self.assertEqual(r["mfa"], "unknown")
        self.assertEqual(weak_label(r)[0], 0)
    def test_root_and_iam_sensitive(self):
        self.assertEqual(weak_label(normalize(event(userIdentity={"type":"Root"})))[0], 1)
        self.assertEqual(weak_label(normalize(event(eventName="CreateAccessKey", eventSource="iam.amazonaws.com")))[0], 1)
    def test_bad_timestamp(self):
        with self.assertRaises(ValueError): normalize(event(eventTime="not-a-date"))
        with self.assertRaises(ValueError): normalize(event(eventTime="2024-01-01T00:00:00"))
    def test_future_values_do_not_mutate_encoder(self):
        train = [normalize(event())]
        enc = fit_encoder(train)
        before = copy.deepcopy(enc)
        future = normalize(event(eventName="FutureAPI", sourceIPAddress="198.51.100.2"))
        arr = transform([future], enc)
        self.assertEqual(enc, before)
        self.assertEqual(arr.shape, (1,len(feature_names(enc))))
        self.assertTrue(np.isfinite(arr).all())
        self.assertNotIn(TARGET, feature_names(enc))
    def test_training_serving_normalization(self):
        a = normalize(event())
        b = normalize(copy.deepcopy(event()))
        enc = fit_encoder([a])
        np.testing.assert_array_equal(transform([a],enc),transform([b],enc))

if __name__ == "__main__": unittest.main()
