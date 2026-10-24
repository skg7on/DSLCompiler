import unittest

from arm_sme_evidence_common import (
    has_sme_instructions, parse_hex_identity)


class ArmSMEEvidenceParserTest(unittest.TestCase):
    def test_sme_instructions_are_required(self):
        sme = """
        0: 2598e3e0  ptrue p0.s
        4: 80812000  fmopa za0.s, p0/m, p1/m, z0.s, z1.s
        8: c00800ff  zero {za}
        c: d503457f  smstart za
        """
        self.assertTrue(has_sme_instructions(sme))

    def test_neon_and_scalar_objects_are_not_sme(self):
        scalar = """
        0: d503201f  nop
        4: 8b010000  add x0, x0, x1
        8: 4e21d800  fadd v0.4s, v0.4s, v1.4s
        c: 4e21d800  fadd v0.4s, v0.4s, v1.4s
        """
        self.assertFalse(has_sme_instructions(scalar))
        # A vector contraction without ZA enablement is not enough.
        outer = "0: 80812000  fmopa za0.s, p0/m, p1/m, z0.s, z1.s"
        self.assertFalse(has_sme_instructions(outer))

    def test_hex_identity_parses_both_forms(self):
        self.assertEqual(parse_hex_identity("deadbeef"), 0xDEADBEEF)
        self.assertEqual(parse_hex_identity("0x10"), 0x10)
        self.assertEqual(parse_hex_identity("0"), 0)
        self.assertEqual(parse_hex_identity("0x0"), 0)

    def test_absent_or_malformed_identity_is_rejected(self):
        for value in (None, "", "not-a-hash", 17):
            self.assertIsNone(parse_hex_identity(value))


if __name__ == "__main__":
    unittest.main()
