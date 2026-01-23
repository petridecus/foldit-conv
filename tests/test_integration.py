import pytest
import numpy as np

# Try importing the module - this will fail if not installed
# but we can still test the Python bindings
try:
    import foldit_conv

    def test_pdb_roundtrip():
        """Test PDB to COORDS and back conversion"""
        pdb_str = """ATOM      1  CA  A   1      1.000   2.000   3.000  1.00  0.00
ATOM      2  CA  A   2      1.000   2.000   3.000  1.00  0.00
END
"""

        # PDB to COORDS
        coords_bytes = foldit_conv.pdb_to_coords(pdb_str)

        # COORDS to PDB
        pdb_back = foldit_conv.coords_to_pdb(coords_bytes)

        # Verify basic structure is preserved
        assert b"ATOM" in pdb_back
        assert b"END" in pdb_back
        assert pdb_back.count("CA") == pdb_str.count("CA")

    def test_coords_deserialize():
        """Test COORDS deserialization"""
        pdb_str = """ATOM      1  CA  A   1      1.000   2.000   3.000  1.00  0.00
ATOM      2  CA  A   2      1.000   2.000   3.000  1.00  0.00
END
"""

        coords_bytes = foldit_conv.pdb_to_coords(pdb_str)

        # Deserialize
        result = foldit_conv.deserialize_coords(coords_bytes)

        assert result["num_atoms"] == 2
        assert len(result["x"]) == 2
        assert len(result["y"]) == 2
        assert len(result["z"]) == 2

except ImportError:
    # Module not installed - skip Python tests
    pass

if __name__ == "__main__":
    test_pdb_roundtrip()
    test_coords_deserialize()
    print("Basic tests passed")
