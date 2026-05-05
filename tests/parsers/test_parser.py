from unittest.mock import MagicMock, patch

from nomad.datamodel import EntryArchive

from nomad_measurements_afm.parsers.parser import NTMDTAFMParser
from nomad_measurements_afm.schema_packages.schema_package import ELNAFMMicroscopy


def test_is_mainfile_valid(tmp_path):
    """Tests if the parser correctly accepts a valid .mdt file."""
    # FIX: Pass the regex rule so the base MatchingParser doesn't reject it
    parser = NTMDTAFMParser(mainfile_name_re=r'^.*\.mdt$')
    valid_file = tmp_path / 'valid_scan.mdt'

    valid_file.write_bytes(b'\x01\xb0\x93\xff\x00\x00\x00\x00')

    with open(valid_file, 'rb') as f:
        valid_buffer = f.read(4)

    assert (
        parser.is_mainfile(
            filename=str(valid_file),
            mime='application/octet-stream',
            buffer=valid_buffer,
            decoded_buffer='',
        )
        is True
    )


def test_is_mainfile_invalid_extension(tmp_path):
    """Tests if the parser rejects files with the wrong extension, even with valid bytes."""
    parser = NTMDTAFMParser(mainfile_name_re=r'^.*\.mdt$')
    invalid_ext = tmp_path / 'invalid_scan.txt'
    invalid_ext.write_bytes(b'\x01\xb0\x93\xff\x00\x00\x00\x00')

    with open(invalid_ext, 'rb') as f:
        valid_buffer = f.read(4)

    assert (
        parser.is_mainfile(
            filename=str(invalid_ext),
            mime='text/plain',
            buffer=valid_buffer,
            decoded_buffer='',
        )
        is False
    )


def test_is_mainfile_invalid_signature(tmp_path):
    """Tests if the parser rejects .mdt files that don't have the correct hex signature."""
    parser = NTMDTAFMParser(mainfile_name_re=r'^.*\.mdt$')
    invalid_sig = tmp_path / 'invalid_sig.mdt'
    invalid_sig.write_bytes(b'abcd\x00\x00\x00\x00')

    with open(invalid_sig, 'rb') as f:
        bad_buffer = f.read(4)

    assert (
        parser.is_mainfile(
            filename=str(invalid_sig),
            mime='application/octet-stream',
            buffer=bad_buffer,
            decoded_buffer='',
        )
        is False
    )


@patch(
    'nomad_measurements_afm.schema_packages.schema_package.ELNAFMMicroscopy.normalize'
)
def test_parse_triggers_schema(mock_normalize):
    """Tests if the parser successfully creates the schema entry and triggers normalization."""
    parser = NTMDTAFMParser(mainfile_name_re=r'^.*\.mdt$')
    archive = EntryArchive()
    archive.m_context = MagicMock()

    parser.parse(mainfile='/fake/path/to/scan_file.mdt', archive=archive)

    assert isinstance(archive.data, ELNAFMMicroscopy)
    assert archive.data.data_file == 'scan_file.mdt'

    mock_normalize.assert_called_once()
