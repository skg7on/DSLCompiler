import unittest

from arm_sve_evidence import has_sve_instructions, parse_hex_identity


class ArmSVEEvidenceParserTest(unittest.TestCase):
    def test_scalable_instructions_are_required(self):
        # A representative SVE disassembly line with a scalable register and
        # one of the SVE-only predicate/load mnemonics.
        sve = """
        0:  25 8c 40 a5  whilelo p0.s, x1, x0
        4:  41 20 c0 85  ld1w z0.s, p0/z, [x2, x1, lsl #2]
        8:  00 00 00 04  fadd z0.s, z0.s, z1.s
        """
        self.assertTrue(has_sve_instructions(sve))

    def test_scalar_and_neon_objects_are_not_sve(self):
        scalar = """
        0:  d503201f  nop
        4:  8b010000  add x0, x0, x1
        8:  4e21d800  fadd v0.4s, v0.4s, v1.4s
        """
        self.assertFalse(has_sve_instructions(scalar))

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
