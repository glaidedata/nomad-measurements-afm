from unittest.mock import MagicMock, patch

from nomad.datamodel import EntryArchive

from nomad_measurements_afm.parsers.parser import AFMParser
from nomad_measurements_afm.schema_packages.schema_package import (
    ELNBrukerMicroscopy,
    ELNNTMDTMicroscopy,
)


def test_is_mainfile_valid_ntmdt(tmp_path):
    """Tests if the parser correctly accepts a valid NT-MDT .mdt file."""
    parser = AFMParser(mainfile_name_re=r'^.*\.(mdt|\d{3})$')
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


def test_is_mainfile_valid_bruker(tmp_path):
    """Tests if the parser correctly accepts a valid Bruker .003 file."""
    parser = AFMParser(mainfile_name_re=r'^.*\.(mdt|\d{3})$')
    valid_file = tmp_path / 'valid_scan.003'

    # Bruker files must start with this exact ASCII string
    valid_file.write_bytes(b'\\*File list\n\\Version: 0x08150307')

    with open(valid_file, 'rb') as f:
        valid_buffer = f.read(50)

    assert (
        parser.is_mainfile(
            filename=str(valid_file),
            mime='application/octet-stream',
            buffer=valid_buffer,
            decoded_buffer='',
        )
        is True
    )


def test_is_mainfile_invalid_signature(tmp_path):
    """Tests if the parser rejects files with the wrong bytes, even if the extension matches."""
    parser = AFMParser(mainfile_name_re=r'^.*\.(mdt|\d{3})$')

    # Invalid NT-MDT
    invalid_mdt = tmp_path / 'invalid_sig.mdt'
    invalid_mdt.write_bytes(b'abcd\x00\x00\x00\x00')
    with open(invalid_mdt, 'rb') as f:
        assert not parser.is_mainfile(
            str(invalid_mdt), 'application/octet-stream', f.read(4), ''
        )

    # Invalid Bruker
    invalid_bruker = tmp_path / 'invalid_sig.003'
    invalid_bruker.write_bytes(b'Random ASCII text that is not a Bruker header')
    with open(invalid_bruker, 'rb') as f:
        assert not parser.is_mainfile(
            str(invalid_bruker), 'application/octet-stream', f.read(50), ''
        )


@patch(
    'nomad_measurements_afm.schema_packages.schema_package.ELNNTMDTMicroscopy.normalize'
)
def test_parse_triggers_ntmdt_schema(mock_normalize):
    """Tests if parsing a .mdt file correctly routes to the NT-MDT schema."""
    parser = AFMParser(mainfile_name_re=r'^.*\.(mdt|\d{3})$')
    archive = EntryArchive()
    archive.m_context = MagicMock()

    parser.parse(mainfile='/fake/path/to/scan_file.mdt', archive=archive)

    assert isinstance(archive.data, ELNNTMDTMicroscopy)
    assert archive.data.data_file == 'scan_file.mdt'
    mock_normalize.assert_called_once()


@patch(
    'nomad_measurements_afm.schema_packages.schema_package.ELNBrukerMicroscopy.normalize'
)
def test_parse_triggers_bruker_schema(mock_normalize):
    """Tests if parsing a .003 file correctly routes to the Bruker schema."""
    parser = AFMParser(mainfile_name_re=r'^.*\.(mdt|\d{3})$')
    archive = EntryArchive()
    archive.m_context = MagicMock()

    parser.parse(mainfile='/fake/path/to/scan_file.003', archive=archive)

    assert isinstance(archive.data, ELNBrukerMicroscopy)
    assert archive.data.data_file == 'scan_file.003'
    mock_normalize.assert_called_once()
