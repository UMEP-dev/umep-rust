"""
SOLWEIG Processing Provider

Registers all SOLWEIG algorithms with the QGIS Processing framework.
"""

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon


class SolweigProvider(QgsProcessingProvider):
    """
    QGIS Processing provider for SOLWEIG algorithms.

    Groups algorithms into categories:
    - Preprocessing: SVF computation
    - Calculation: Tmrt (single, timeseries, tiled)
    - Post-processing: UTCI, PET
    - Utilities: EPW import
    """

    def id(self):
        """Unique provider ID used in processing scripts."""
        return "solweig"

    def name(self):
        """Display name shown in Processing Toolbox."""
        return "SOLWEIG"

    def longName(self):
        """Extended name for provider description."""
        return "SOLWEIG - Solar and Longwave Environmental Irradiance Geometry"

    def icon(self):
        """Provider icon shown in Processing Toolbox."""
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QgsProcessingProvider.icon(self)

    def loadAlgorithms(self):
        """
        Load and register all SOLWEIG algorithms.

        Called by QGIS when the provider is initialized.
        """
        # Preprocessing algorithms
        from .algorithms.preprocess.svf_preprocessing import SvfPreprocessingAlgorithm

        self.addAlgorithm(SvfPreprocessingAlgorithm())

        # Calculation algorithms
        from .algorithms.calculation.single_timestep import SingleTimestepAlgorithm
        from .algorithms.calculation.tiled_processing import TiledProcessingAlgorithm
        from .algorithms.calculation.timeseries import TimeseriesAlgorithm

        self.addAlgorithm(SingleTimestepAlgorithm())
        self.addAlgorithm(TimeseriesAlgorithm())
        self.addAlgorithm(TiledProcessingAlgorithm())

        # Post-processing algorithms
        from .algorithms.postprocess.pet import PetAlgorithm
        from .algorithms.postprocess.utci import UtciAlgorithm

        self.addAlgorithm(UtciAlgorithm())
        self.addAlgorithm(PetAlgorithm())

        # Utility algorithms
        from .algorithms.utilities.epw_import import EpwImportAlgorithm

        self.addAlgorithm(EpwImportAlgorithm())
