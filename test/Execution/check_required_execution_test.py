import unittest

from check_required_execution import parse_outcomes, validate


class RequiredExecutionEvidenceTest(unittest.TestCase):
    def test_all_required_cases_must_pass(self):
        report = """<testsuites><testsuite>
          <testcase name="MappedAVX2Acceptance" />
          <testcase name="MappedAllocationLifetimeTest" />
        </testsuite></testsuites>"""
        self.assertIsNone(validate(parse_outcomes(report)))

    def test_missing_skipped_failed_and_empty_reports_fail(self):
        self.assertIn("missing", validate({}))
        for outcome in ("skipped", "failed"):
            outcomes = {"MappedAVX2Acceptance": outcome,
                        "MappedAllocationLifetimeTest": "passed"}
            self.assertIn("did not pass", validate(outcomes))
        self.assertIn("missing", validate({"MappedAVX2Acceptance": "passed"}))

    def test_junit_skip_and_failure_are_not_passes(self):
        report = """<testsuites><testsuite>
          <testcase name="MappedAVX2Acceptance"><skipped /></testcase>
          <testcase name="MappedAllocationLifetimeTest"><failure /></testcase>
        </testsuite></testsuites>"""
        outcomes = parse_outcomes(report)
        self.assertEqual(outcomes["MappedAVX2Acceptance"], "skipped")
        self.assertEqual(outcomes["MappedAllocationLifetimeTest"], "failed")
        self.assertIn("did not pass", validate(outcomes))


if __name__ == "__main__":
    unittest.main()
