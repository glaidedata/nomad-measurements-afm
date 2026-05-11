import datetime
from unittest.mock import MagicMock, patch

import numpy as np
from nomad.datamodel import EntryArchive
from nomad.datamodel.datamodel import EntryMetadata

from nomad_measurements_afm.schema_packages.schema_package import (
    ELNBrukerMicroscopy,
    ELNNTMDTMicroscopy,
)

# ==========================================
# 1. NT-MDT SCHEMA TEST
# ==========================================


@patch('nomad_measurements_afm.schema_packages.schema_package.read_ntmdt')
def test_ntmdt_schema_normalization(mock_read_ntmdt):
    """Tests if the NT-MDT schema accurately maps data into NOMAD Quantities."""

    mock_data = MagicMock()
    mock_data.metadata = {'Total Frames': 1}

    mock_channel = MagicMock()
    mock_channel.metadata = {
        'channel_index': 'Height',
        'x_resolution': 256,
        'y_resolution': 256,
        'velocity': 5.2,
        'setpoint': 4.9,
        'bias_voltage': 0.5,
        'x_scale': {'step': 117.67, 'unit': 'angstrom'},
        'y_scale': {'step': 117.67, 'unit': 'angstrom'},
        'z_scale': {'step': 3.48e-06, 'unit': 'micro_meter'},
        'xml_metadata': (
            '<Probe><Material>Si</Material><ResFreq>150</ResFreq><TipRadius>10</TipRadius></Probe>'
            '<Common><Temperature>23.11</Temperature><Humidity>61.56</Humidity>'
            '<BuildID>Nova_SPM 1.2.</BuildID><Technic>SemiContact</Technic></Common>'
            '<Scan><ReadyPointsZ>2</ReadyPointsZ></Scan>'
        ),
    }
    mock_channel.data = np.random.rand(256, 256)

    mock_data.channels = {'1F:Phase1': mock_channel}
    mock_read_ntmdt.return_value = mock_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='dummy_scan.mdt')

    mock_file_context = MagicMock()
    mock_file_context.name = 'dummy_scan.mdt'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file_context

    entry = ELNNTMDTMicroscopy()
    entry.data_file = 'dummy_scan.mdt'

    entry.normalize(archive, None)

    assert entry.instrument_model == 'NT-MDT AFM'
    assert entry.total_frames == 1
    assert entry.software_version == 'Nova_SPM 1.2.'

    assert entry.probe_setup.resonant_frequency.magnitude == 150.0  # noqa: PLR2004
    assert entry.acquisition_setup.environment_temperature.magnitude == 23.11  # noqa: PLR2004
    assert entry.results[0].channels[0].channel_name == '1F:Phase1'


# ==========================================
# 2. BRUKER SCHEMA TEST
# ==========================================


@patch('nomad_measurements_afm.schema_packages.schema_package.read_bruker')
def test_bruker_schema_normalization(mock_read_bruker):
    """Tests if the Bruker schema accurately maps deep metadata and channels."""

    mock_data = MagicMock()
    mock_data.metadata = {
        'instrument_model': 'MultiMode 8',
        'operating_mode': 'PeakForce QNM',
        'Date': '05:05:28 PM Thu Feb 26 2026',
        'Version': '0x08150307',
        'Scanner file': '9575jvlr.scn',
        'Piezo size': 'J',
        'Medium': 'Fluid',
        'X Offset': '-11892.6 nm',
        'Y Offset': '1484.38 nm',
        'Rotate Ang.': '90.5',
        'Z Range': '10.8798',
        'IntGain': '0.5',
        'PrpGain': '0.5',
        'Engage Setpoint': '0.85',
        'Peak Force Amplitude': '100',
        'Peak Force Engage Setpoint': '0.15',
        "Sample Poisson's Ratio": '0.3',
        'Sync Distance': '82',
        'scan_size': 10000.0,
        'scan_size_unit': 'nm',
        'scan_rate': 0.542535,
    }

    mock_channel = MagicMock()
    mock_channel.metadata = {'x_res': 256, 'y_res': 256, 'channel_name': 'Height'}
    mock_channel.data = np.random.rand(256, 256)
    mock_data.channels = {'Retrace_Height': mock_channel}

    mock_read_bruker.return_value = mock_data

    archive = EntryArchive()
    archive.m_context = MagicMock()
    archive.metadata = EntryMetadata(entry_name='bruker_scan.003')

    mock_file_context = MagicMock()
    mock_file_context.name = 'bruker_scan.003'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file_context

    entry = ELNBrukerMicroscopy()
    entry.data_file = 'bruker_scan.003'

    entry.normalize(archive, None)

    # Assert Base Metadata
    assert entry.instrument_model == 'MultiMode 8'
    assert entry.measurement_technique == 'PeakForce QNM'
    assert entry.datetime == datetime.datetime(
        2026, 2, 26, 17, 5, 28, tzinfo=datetime.timezone.utc
    )

    # Assert Physical Math (10,000 nm should convert to 0.00001 m)
    assert np.isclose(entry.acquisition_setup.scan_size.magnitude, 10000.0 * 1e-9)
    assert entry.acquisition_setup.scan_rate.magnitude == 0.542535  # noqa: PLR2004

    # Assert the Bruker Specific Setup mapped perfectly
    assert entry.bruker_setup.software_version == '0x08150307'
    assert entry.bruker_setup.scanner_file == '9575jvlr.scn'
    assert entry.bruker_setup.medium == 'Fluid'

    # Assert string-to-float physical conversion
    assert np.isclose(entry.bruker_setup.x_offset.magnitude, -11892.6 * 1e-9)
    assert entry.bruker_setup.scan_angle.magnitude == 90.5  # noqa: PLR2004
    assert entry.bruker_setup.peak_force_amplitude == 100.0  # noqa: PLR2004

    # Assert the dictionary catch-all
    assert entry.bruker_setup.raw_metadata['Piezo size'] == 'J'

    # Assert Channel Data (Step size = scan_size / resolution)
    channel = entry.results[0].channels[0]
    assert channel.channel_name == 'Retrace_Height'
    expected_step = (10000.0 * 1e-9) / 256
    assert np.isclose(channel.x_step_size.magnitude, expected_step)
