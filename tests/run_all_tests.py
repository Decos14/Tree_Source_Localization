import unittest
import os

def main():
    os.chdir(os.path.dirname(__file__))  # Ensure working directory is script's location

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load tests from both test files
    unit_tests = loader.discover('.', pattern='*_Unit_Tests.py')
    regression_tests = loader.discover('.', pattern='*_Regression_Tests.py')

    # Combine them into one suite
    suite.addTests(unit_tests)
    suite.addTests(regression_tests)

    # Run combined suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    main()
