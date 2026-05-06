from unittest.mock import MagicMock, patch

import numpy as np
from nomad.datamodel import EntryArchive
from nomad.datamodel.datamodel import EntryMetadata  # <-- Added this import

from nomad_measurements_afm.schema_packages.schema_package import ELNAFMMicroscopy


@patch('nomad_measurements_afm.schema_packages.schema_package.read_ntmdt')
def test_afm_schema_normalization(mock_read_ntmdt):
    """Tests if the schema accurately maps the extracted reader data into NOMAD Quantities."""

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

    # FIX: Use actual EntryMetadata instead of a MagicMock to silence the warning
    archive.metadata = EntryMetadata(entry_name='dummy_scan.mdt')

    mock_file_context = MagicMock()
    mock_file_context.name = 'dummy_scan.mdt'
    archive.m_context.raw_file.return_value.__enter__.return_value = mock_file_context

    entry = ELNAFMMicroscopy()
    entry.data_file = 'dummy_scan.mdt'

    entry.normalize(archive, None)

    assert entry.instrument_model == 'NT-MDT AFM'
    assert entry.total_frames == 1
    assert entry.software_version == 'Nova_SPM 1.2.'
    assert entry.measurement_technique == 'SemiContact'

    assert entry.probe_setup is not None
    assert entry.probe_setup.material == 'Si'

    # FIX: Add .magnitude to any Quantity that has a unit assigned in the schema
    assert entry.probe_setup.resonant_frequency.magnitude == 150.0  # noqa: PLR2004
    assert entry.probe_setup.tip_radius.magnitude == 10.0  # noqa: PLR2004

    assert entry.acquisition_setup is not None
    assert entry.acquisition_setup.environment_temperature.magnitude == 23.11  # noqa: PLR2004
    assert (
        entry.acquisition_setup.environment_humidity == 61.56  # noqa: PLR2004
    )  # No unit in schema
    assert entry.acquisition_setup.scan_velocity.magnitude == 5.2  # noqa: PLR2004
    assert entry.acquisition_setup.setpoint == 4.9  # No unit in schema # noqa: PLR2004
    assert entry.acquisition_setup.bias_voltage.magnitude == 0.5  # noqa: PLR2004
    assert entry.acquisition_setup.scan_direction == 'Forward'

    assert len(entry.results) == 1
    assert len(entry.results[0].channels) == 1
    channel = entry.results[0].channels[0]

    assert channel.channel_name == '1F:Phase1'
    assert channel.channel_type == 'Height'
    assert channel.x_resolution == 256  # noqa: PLR2004
    assert channel.y_resolution == 256  # noqa: PLR2004
    assert channel.z_resolution == 2  # noqa: PLR2004

    assert np.isclose(channel.x_step_size.magnitude, 117.67 * 1e-10)
    assert np.isclose(channel.y_step_size.magnitude, 117.67 * 1e-10)
    assert channel.z_step_size == 3.48e-06  # No unit in schema # noqa: PLR2004

    np.testing.assert_array_equal(channel.data, mock_channel.data)
