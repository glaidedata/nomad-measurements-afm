import re
from typing import TYPE_CHECKING, Dict, Any
import numpy as np
from nomad.datamodel.data import ArchiveSection, EntryData, JSON
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum
from nomad.datamodel.metainfo.basesections import Measurement, MeasurementResult
from nomad.metainfo import Quantity, SchemaPackage, Section, SubSection
import datetime

# Import both readers!
from readers_ientrance import read_ntmdt, read_bruker

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

m_package = SchemaPackage()


# ==========================================
# 1. SHARED AFM INSTRUMENT & SETUP
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
    scan_rate = Quantity(
        type=np.float64,
        unit='Hz',
        description='Number of scan lines performed per second.'
    )
    scan_size = Quantity(
        type=np.float64,
        unit='m',
        description='Total physical size of the scan window.'
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
        description='Direction of the scan trace (e.g., Forward, Retrace).'
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
# 2. SHARED AFM RESULTS (THE SCANS)
# ==========================================

class AFMChannel(ArchiveSection):
    """A repeating section to hold each individual 2D scan and its metadata."""
    channel_name = Quantity(
        type=str,
        description='Name of the extracted signal (e.g., 1F:Phase1, Retrace_Height).'
    )
    channel_type = Quantity(
        type=str,
        description='Classification of the signal (e.g., topography, phase, amplitude).'
    )

    x_resolution = Quantity(type=np.int32, description='Number of pixels in the X direction.')
    y_resolution = Quantity(type=np.int32, description='Number of pixels in the Y direction.')
    z_resolution = Quantity(type=np.int32, description='Number of data points in the Z direction.')

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

    data = Quantity(
        type=np.float64,
        shape=['*', '*'],
        description='The raw 2D NumPy array of the AFM scan.'
    )

class AFMResult(MeasurementResult):
    channels = SubSection(section_def=AFMChannel, repeats=True)


# ==========================================
# 3. BASE AFM ENTRY
# ==========================================

class BaseAFMMicroscopy(Measurement):
    """Base class containing shared attributes for all AFM entries."""
    data_file = Quantity(
        type=str,
        a_eln=dict(component=ELNComponentEnum.FileEditQuantity),
        a_browser=dict(adaptor='RawFileAdaptor'),
        description='The raw data file.',
    )

    instrument_model = Quantity(
        type=str,
        description='The model of the AFM instrument.',
        a_eln=dict(component=ELNComponentEnum.StringEditQuantity)
    )

    software_version = Quantity(
        type=str,
        description='Software used to record the scan.',
        a_eln=dict(component=ELNComponentEnum.StringEditQuantity)
    )

    measurement_technique = Quantity(
        type=str,
        description='The specific technique used.',
        a_eln=dict(component=ELNComponentEnum.StringEditQuantity)
    )

    probe_setup = SubSection(section_def=AFMProbe)
    acquisition_setup = SubSection(section_def=AFMAcquisitionSetup)
    results = SubSection(section_def=AFMResult, repeats=True)


# ==========================================
# 4. NT-MDT SPECIFIC SCHEMA
# ==========================================

class ELNNTMDTMicroscopy(BaseAFMMicroscopy, EntryData):
    m_def = Section(
        label='NT-MDT Atomic Force Microscopy',
        a_eln=dict(lane_width='600px'),
        a_template=dict(measurement_identifiers=dict()),
    )

    total_frames = Quantity(
        type=np.int32,
        description='Total number of channels/scans in the file.',
        a_eln=dict(component=ELNComponentEnum.NumberEditQuantity)
    )

    def _extract_from_xml(self, xml_string: str, tag: str, cast_type=str):
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

    def _populate_global_setup(self, meta: Dict[str, Any], xml: str, direction: str):
        self.software_version = self._extract_from_xml(xml, 'BuildID')
        self.measurement_technique = self._extract_from_xml(xml, 'Technic')

        self.probe_setup.material = self._extract_from_xml(xml, 'Material')
        self.probe_setup.shape = self._extract_from_xml(xml, 'Shape')
        self.probe_setup.probe_id = self._extract_from_xml(xml, 'ID')
        self.probe_setup.stiffness = self._extract_from_xml(xml, 'Stiffness', float)
        self.probe_setup.tip_radius = self._extract_from_xml(xml, 'TipRadius', float)
        self.probe_setup.resonant_frequency = self._extract_from_xml(xml, 'ResFreq', float)

        self.acquisition_setup.environment_temperature = self._extract_from_xml(xml, 'Temperature', float)
        self.acquisition_setup.environment_humidity = self._extract_from_xml(xml, 'Humidity', float)

        self.acquisition_setup.scan_velocity = meta.get('velocity')
        self.acquisition_setup.setpoint = meta.get('setpoint')
        self.acquisition_setup.bias_voltage = meta.get('bias_voltage')
        self.acquisition_setup.scan_direction = direction

    def _create_channel(self, name: str, meta: Dict[str, Any], xml: str, data: np.ndarray) -> AFMChannel:
        x_scale_dict = meta.get('x_scale', {})
        y_scale_dict = meta.get('y_scale', {})
        z_scale_dict = meta.get('z_scale', {})

        x_step = x_scale_dict.get('step')
        if x_step and x_scale_dict.get('unit') == 'angstrom':
            x_step *= 1e-10

        y_step = y_scale_dict.get('step')
        if y_step and y_scale_dict.get('unit') == 'angstrom':
            y_step *= 1e-10

        z_res_val = self._extract_from_xml(xml, 'ReadyPointsZ', int) if xml else None

        return AFMChannel(
            channel_name=name,
            channel_type=meta.get('channel_index'),
            x_resolution=meta.get('x_resolution'),
            y_resolution=meta.get('y_resolution'),
            z_resolution=z_res_val,
            x_step_size=x_step,
            y_step_size=y_step,
            z_step_size=z_scale_dict.get('step'),
            data=data
        )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if not self.data_file:
            super().normalize(archive, logger)
            return

        try:
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

            global_setup_populated = False
            channel_sections = []

            if afm_data.channels:
                for name, channel_obj in afm_data.channels.items():
                    meta = channel_obj.metadata
                    xml = meta.get('xml_metadata', '')

                    direction = 'Forward' if name.startswith('1F:') else 'Backward' if name.startswith('1B:') else 'Unknown'

                    if not global_setup_populated and xml:
                        self._populate_global_setup(meta, xml, direction)
                        global_setup_populated = True

                    channel = self._create_channel(name, meta, xml, channel_obj.data)
                    channel_sections.append(channel)

            self.results[0].channels = channel_sections

        except Exception as e:
            if logger:
                logger.error(f'Error parsing NT-MDT file: {e}')
            raise e

        super().normalize(archive, logger)


# ==========================================
# 5. BRUKER SPECIFIC SCHEMA
# ==========================================

class BrukerSpecificSetup(ArchiveSection):
    """Metadata specific to Bruker/Nanoscope AFMs."""
    # --- Hardware & Environment ---
    software_version = Quantity(
        type=str,
        description='Nanoscope software version.'
    )
    scanner_file = Quantity(
        type=str,
        description='The specific scanner calibration file used.'
    )
    piezo_size = Quantity(
        type=str,
        description='The hardware size classification of the piezo scanner (e.g., J, E).'
    )
    medium = Quantity(
        type=str,
        description='The imaging environment (e.g., Air, Fluid).'
    )

    # --- Scan Geometry ---
    x_offset = Quantity(
        type=np.float64,
        unit='m',
        description='X-axis offset of the scan window from the piezo center.'
    )
    y_offset = Quantity(
        type=np.float64,
        unit='m',
        description='Y-axis offset of the scan window from the piezo center.'
    )
    scan_angle = Quantity(
        type=np.float64,
        unit='degree',
        description='Rotation angle of the scan.'
    )
    z_range = Quantity(
        type=np.float64,
        description='Maximum allowed vertical extension of the Z piezo.'
    )

    # --- PID Feedback ---
    integral_gain = Quantity(
        type=np.float64,
        description='Main integral gain for the feedback loop.'
    )
    proportional_gain = Quantity(
        type=np.float64,
        description='Main proportional gain for the feedback loop.'
    )
    engage_setpoint = Quantity(
        type=np.float64,
        description='The threshold value used to trigger surface engagement.'
    )

    # --- PeakForce QNM & Nanomechanics ---
    operating_mode = Quantity(
        type=str,
        description='e.g., PeakForce QNM, Tapping.'
    )
    peak_force_amplitude = Quantity(
        type=np.float64,
        description='Oscillation amplitude for PeakForce Tapping mode.'
    )
    peak_force_engage_setpoint = Quantity(
        type=np.float64,
        description='Engage setpoint specific to PeakForce mode.'
    )
    sync_distance = Quantity(
        type=np.float64,
        description='Phase delay distance between the drive signal and tip response.'
    )

    # --- Advanced Mechanical Models ---
    sample_poissons_ratio = Quantity(type=np.float64)
    tip_half_angle = Quantity(type=np.float64, description='Tip angle in radians or degrees.')
    modulus_fit_model = Quantity(type=str, description='e.g., Hertzian (Spherical).')
    scan_asyst_noise_threshold = Quantity(type=np.float64)
    torsional_frequency = Quantity(type=np.float64, description='Torsional resonance.')
    torsional_q_factor = Quantity(type=np.float64)

    # ==========================================
    # THE CATCH-ALL FOR HEAVILY DETAILED HEADER
    # ==========================================
    raw_metadata = Quantity(
        type=JSON,
        description='A complete dictionary dump of every single line in the Bruker header.',
    )


class ELNBrukerMicroscopy(BaseAFMMicroscopy, EntryData):
    m_def = Section(
        label='Bruker Nanoscope AFM',
        a_eln=dict(lane_width='600px'),
        a_template=dict(measurement_identifiers=dict()),
    )

    bruker_setup = SubSection(section_def=BrukerSpecificSetup)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger'):
        if not self.data_file:
            super().normalize(archive, logger)
            return

        try:
            with archive.m_context.raw_file(self.data_file) as file:
                afm_data = read_bruker(file.name)

            # --- 1. Map Global Base Setup ---
            self.instrument_model = afm_data.metadata.get("instrument_model", "Bruker AFM")
            self.measurement_technique = afm_data.metadata.get("operating_mode")

            # Parse the timestamp directly into NOMAD's core datetime field
            date_str = afm_data.metadata.get("Date")
            if date_str:
                try:
                    # Bruker format: "05:05:28 PM Thu Feb 26 2026"
                    self.datetime = datetime.datetime.strptime(date_str, "%I:%M:%S %p %a %b %d %Y")
                except ValueError:
                    pass

            if not self.results:
                self.results = [AFMResult()]
            if not self.probe_setup:
                self.probe_setup = AFMProbe()
            if not self.acquisition_setup:
                self.acquisition_setup = AFMAcquisitionSetup()
            if not getattr(self, "bruker_setup", None):
                self.bruker_setup = BrukerSpecificSetup()

            self.probe_setup.probe_id = afm_data.metadata.get("probe_id")
            if afm_data.metadata.get("tip_radius"):
                self.probe_setup.tip_radius = afm_data.metadata.get("tip_radius")
            if afm_data.metadata.get("scan_rate"):
                self.acquisition_setup.scan_rate = afm_data.metadata.get("scan_rate")

            # Convert Scan Size to SI Meters
            scan_size_raw = afm_data.metadata.get("scan_size")
            scan_unit = afm_data.metadata.get("scan_size_unit", "nm")
            if scan_size_raw is not None:
                if scan_unit.lower() == "nm":
                    self.acquisition_setup.scan_size = scan_size_raw * 1e-9
                elif scan_unit.lower() in ["um", "~m"]:
                    self.acquisition_setup.scan_size = scan_size_raw * 1e-6
                else:
                    self.acquisition_setup.scan_size = scan_size_raw

            # --- 2. Map the New Bruker-Specific Setup ---
            # Strings
            self.bruker_setup.software_version = afm_data.metadata.get("Version")
            self.bruker_setup.scanner_file = afm_data.metadata.get("Scanner file")
            self.bruker_setup.piezo_size = afm_data.metadata.get("Piezo size")
            self.bruker_setup.medium = afm_data.metadata.get("Medium")
            self.bruker_setup.operating_mode = afm_data.metadata.get("Operating mode")
            self.bruker_setup.modulus_fit_model = afm_data.metadata.get("Modulus Fit Model")

            # Helper to safely extract and cast floats
            def safe_float(key):
                val = afm_data.metadata.get(key)
                if val is not None:
                    try:
                        # Handles strings like "-11892.6 nm" by splitting at the space
                        return float(val.split()[0]) if isinstance(val, str) else float(val)
                    except ValueError:
                        pass
                return None

            # Physical Conversions (nm to m)
            x_offset = safe_float("X Offset")
            if x_offset is not None: self.bruker_setup.x_offset = x_offset * 1e-9

            y_offset = safe_float("Y Offset")
            if y_offset is not None: self.bruker_setup.y_offset = y_offset * 1e-9

            # Direct floats
            self.bruker_setup.scan_angle = safe_float("Rotate Ang.")
            self.bruker_setup.z_range = safe_float("Z Range")
            self.bruker_setup.integral_gain = safe_float("IntGain")
            self.bruker_setup.proportional_gain = safe_float("PrpGain")
            self.bruker_setup.engage_setpoint = safe_float("Engage Setpoint")
            self.bruker_setup.peak_force_amplitude = safe_float("Peak Force Amplitude")
            self.bruker_setup.peak_force_engage_setpoint = safe_float("Peak Force Engage Setpoint")
            self.bruker_setup.sync_distance = safe_float("Sync Distance")
            self.bruker_setup.sample_poissons_ratio = safe_float("Sample Poisson's Ratio")
            self.bruker_setup.tip_half_angle = safe_float("Tip Half Angle")
            self.bruker_setup.scan_asyst_noise_threshold = safe_float("ScanAsyst Noise Threshold")
            self.bruker_setup.torsional_frequency = safe_float("Torsional Freq")
            self.bruker_setup.torsional_q_factor = safe_float("Torsional Q")

            # The Catch-All!
            self.bruker_setup.raw_metadata = afm_data.metadata

            # --- 3. Map Channels ---
            channel_sections = []
            if afm_data.channels:
                for name, channel_obj in afm_data.channels.items():
                    meta = channel_obj.metadata

                    x_res = meta.get("x_res")
                    y_res = meta.get("y_res")

                    x_step = None
                    y_step = None
                    if self.acquisition_setup.scan_size and x_res and y_res:
                        x_step = self.acquisition_setup.scan_size / x_res
                        y_step = self.acquisition_setup.scan_size / y_res

                    channel = AFMChannel(
                        channel_name=name,
                        channel_type=meta.get("channel_name"),
                        x_resolution=x_res,
                        y_resolution=y_res,
                        x_step_size=x_step,
                        y_step_size=y_step,
                        data=channel_obj.data
                    )
                    channel_sections.append(channel)

            self.results[0].channels = channel_sections

        except Exception as e:
            if logger:
                logger.error(f'Error parsing Bruker file: {e}')
            raise e

        super().normalize(archive, logger)

m_package.__init_metainfo__()