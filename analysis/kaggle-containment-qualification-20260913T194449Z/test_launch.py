import unittest
import launch_once
class Approval(unittest.TestCase):
    def test_approved(self):self.assertTrue(launch_once.approval_valid({'status':'resolved_for_bounded_probe','manifest_sha256':'abc'},'abc'))
    def test_hold(self):self.assertFalse(launch_once.approval_valid({'status':'HOLD','manifest_sha256':'abc'},'abc'))
    def test_missing(self):self.assertFalse(launch_once.approval_valid({'manifest_sha256':'abc'},'abc'))
    def test_wrong_hash(self):self.assertFalse(launch_once.approval_valid({'status':'resolved_for_bounded_probe','manifest_sha256':'xyz'},'abc'))
if __name__=='__main__':unittest.main()
