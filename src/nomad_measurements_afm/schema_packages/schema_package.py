import re
from typing import TYPE_CHECKING
import numpy as np
from nomad.datamodel.data import ArchiveSection, EntryData
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import Measurement, MeasurementResult
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection

from readers_ientrance import read_ntmdt

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

m_package = SchemaPackage()


# ==========================================
# 1. AFM INSTRUMENT & SETUP
# ==========================================

class AFMProbe(ArchiveSection):
    """Details about the physical cantilever/probe used."""
    material = Quantity(type=str, description='Material of the cantilever.')
    shape = Quantity(type=str, description='Geometry of the cantilever.')
    probe_id = Quantity(type=str, description='Commercial ID or model name of the probe.')
    stiffness = Quantity(
        type=np.float64,
        unit='N/m',
        description='Spring constant/stiffness of the cantilever.'
    )
    resonant_frequency = Quantity(
        type=np.float64,
        unit='kHz',
        description='Resonant frequency of the probe in air.'
    )
    tip_radius = Quantity(
        type=np.float64,
        unit='nm',
        description='Radius of curvature of the probe tip.'
    )


class AFMAcquisitionSetup(ArchiveSection):
    """Parameters governing how the scan was executed."""
    scan_velocity = Quantity(
        type=np.float64,
        unit='um/s',
        description='Speed of the tip across the surface.'
    )
    setpoint = Quantity(
        type=np.float64,
        description='Feedback setpoint (units vary by scan mode, e.g., nA, V).'
    )
    bias_voltage = Quantity(
        type=np.float64,
        unit='V',
        description='Bias voltage applied between tip and sample.'
    )
    scan_direction = Quantity(
        type=str,
        description='Direction of the scan trace (e.g., Forward, Backward).'
    )
    environment_temperature = Quantity(
        type=np.float64,
        unit='celsius',
        description='Ambient temperature during the scan.'
    )
    environment_humidity = Quantity(
        type=np.float64,
        description='Ambient relative humidity (%) during the scan.'
    )


# ==========================================
# 2. AFM RESULTS (THE SCANS)
# ==========================================

class AFMChannel(ArchiveSection):
    """A repeating section to hold each individual 2D scan and its metadata."""
    channel_name = Quantity(
        type=str,
        description='Name of the extracted signal (e.g., 1F:Phase1, 1B:Height1).'
    )
    channel_type = Quantity(
        type=str,
        description='Classification of the signal (e.g., topography, phase, amplitude).'
    )

    # Image Dimensions
    x_resolution = Quantity(type=np.int32, description='Number of pixels in the X direction.')
    y_resolution = Quantity(type=np.int32, description='Number of pixels in the Y direction.')
    z_resolution = Quantity(type=np.int32, description='Number of data points in the Z direction.')

    # Physical Scaling (Step sizes from the Kaitai variables)
    x_step_size = Quantity(
        type=np.float64,
        unit='m',
        description='Physical size of one pixel in the X direction.'
    )
    y_step_size = Quantity(
        type=np.float64,
        unit='m',
        description='Physical size of one pixel in the Y direction.'
    )
    z_step_size = Quantity(
        type=np.float64,
        description='Scaling multiplier to convert raw Z values to physical units.'
    )

    # The actual 2D Image Matrix
    data = Quantity(
        type=np.float64,
        shape=['*', '*'],
        description='The raw 2D NumPy array of the AFM scan.'
    )

class AFMResult(MeasurementResult):
    channels = SubSection(section_def=AFMChannel, repeats=True)


# ==========================================
# 3. MAIN AFM SCHEMA
# ==========================================

class ELNAFMMicroscopy(Measurement, EntryData):
    m_def = Section(
        label='NT-MDT Atomic Force Microscopy',
        a_eln=dict(lane_width='600px'),
        a_template=dict(
            measurement_identifiers=dict(),
        ),
    )

    data_file = Quantity(
        type=str,
        a_eln=dict(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The raw .mdt binary data file.',
    )

    instrument_model = Quantity(type=str)
    software_version = Quantity(type=str)
    total_frames = Quantity(type=np.int32)
    measurement_technique = Quantity(type=str)

    # Subsections
    probe_setup = SubSection(section_def=AFMProbe)
    acquisition_setup = SubSection(section_def=AFMAcquisitionSetup)
    results = SubSection(section_def=AFMResult, repeats=True)

    def _extract_from_xml(self, xml_string: str, tag: str, cast_type=str):
        """Helper to safely extract data from NT-MDT's embedded XML tags."""
        if not xml_string:
            return None
        match = re.search(f'<{tag}>(.*?)</{tag}>', xml_string)
        if match:
            val = match.group(1).strip()
            if not val or val == 'Unknown':
                return None
            try:
                return cast_type(val)
            except ValueError:
                return None
        return None

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if not self.data_file:
            super().normalize(archive, logger)
            return

        try:
            # 1. Pass the file to your reader
            with archive.m_context.raw_file(self.data_file) as file:
                afm_data = read_ntmdt(file.name)

            self.instrument_model = "NT-MDT AFM"
            self.total_frames = afm_data.metadata.get("Total Frames")

            if not self.results:
                self.results = [AFMResult()]

            if not self.probe_setup:
                self.probe_setup = AFMProbe()

            if not self.acquisition_setup:
                self.acquisition_setup = AFMAcquisitionSetup()

            # Use the first channel's metadata to populate the global setup
            # (Assuming the probe doesn't magically change in the middle of a scan file)
            global_setup_populated = False

            channel_sections = []

            if afm_data.channels:
                for name, channel_obj in afm_data.channels.items():
                    meta = channel_obj.metadata
                    xml = meta.get('xml_metadata', '')

                    # Determine scan direction from title (e.g., '1F:Phase1')
                    direction = 'Forward' if name.startswith('1F:') else 'Backward' if name.startswith('1B:') else 'Unknown'

                    # --- Populate Global Setup (Only Once) ---
                    if not global_setup_populated and xml:
                        self.software_version = self._extract_from_xml(xml, 'BuildID')
                        self.measurement_technique = self._extract_from_xml(xml, 'Technic')

                        # Extract Probe Data
                        self.probe_setup.material = self._extract_from_xml(xml, 'Material')
                        self.probe_setup.shape = self._extract_from_xml(xml, 'Shape')
                        self.probe_setup.probe_id = self._extract_from_xml(xml, 'ID')
                        self.probe_setup.stiffness = self._extract_from_xml(xml, 'Stiffness', float)
                        self.probe_setup.tip_radius = self._extract_from_xml(xml, 'TipRadius', float)
                        self.probe_setup.resonant_frequency = self._extract_from_xml(xml, 'ResFreq', float)

                        # Extract Environmental Data
                        self.acquisition_setup.environment_temperature = self._extract_from_xml(xml, 'Temperature', float)
                        self.acquisition_setup.environment_humidity = self._extract_from_xml(xml, 'Humidity', float)

                        # Extract Scan Parameters
                        self.acquisition_setup.scan_velocity = meta.get('velocity')
                        self.acquisition_setup.setpoint = meta.get('setpoint')
                        self.acquisition_setup.bias_voltage = meta.get('bias_voltage')
                        self.acquisition_setup.scan_direction = direction

                        global_setup_populated = True

                    # --- Extract Individual Channel Data ---

                    # Safely extract scales
                    x_scale_dict = meta.get('x_scale', {})
                    y_scale_dict = meta.get('y_scale', {})
                    z_scale_dict = meta.get('z_scale', {})

                    # Convert angstroms to meters for standard SI storage
                    x_step = x_scale_dict.get('step')
                    if x_step and x_scale_dict.get('unit') == 'angstrom':
                        x_step *= 1e-10

                    y_step = y_scale_dict.get('step')
                    if y_step and y_scale_dict.get('unit') == 'angstrom':
                        y_step *= 1e-10

                    # Extract z_resolution from XML if available
                    z_res_val = self._extract_from_xml(xml, 'ReadyPointsZ', int) if xml else None

                    channel = AFMChannel(
                        channel_name=name,
                        channel_type=meta.get('channel_index'),
                        x_resolution=meta.get('x_resolution'),
                        y_resolution=meta.get('y_resolution'),
                        z_resolution=z_res_val,
                        x_step_size=x_step,
                        y_step_size=y_step,
                        z_step_size=z_scale_dict.get('step'),
                        data=channel_obj.data
                    )
                    channel_sections.append(channel)

            # Attach the processed channels to the results
            self.results[0].channels = channel_sections

        except Exception as e:
            if logger:
                logger.error(f'Error parsing NT-MDT file: {e}')
            raise e

        super().normalize(archive, logger)

m_package.__init_metainfo__()