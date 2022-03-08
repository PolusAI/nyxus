import nyxus
import unittest

class TestNyxus(unittest.TestCase):

    def test_import_module(self):
        self.assertEqual(nyxus.__name__ , "nyxus")