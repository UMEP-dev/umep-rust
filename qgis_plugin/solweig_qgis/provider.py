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

    Algorithms:
    - Preprocessing: SVF computation
    - Calculation: Unified SOLWEIG calculation (single/timeseries/tiled + UTCI/PET)
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
        # Preprocessing
        from .algorithms.preprocess.svf_preprocessing import SvfPreprocessingAlgorithm

        self.addAlgorithm(SvfPreprocessingAlgorithm())

        # Main calculation (unified: single/timeseries/tiled + UTCI/PET)
        from .algorithms.calculation.solweig_calculation import SolweigCalculationAlgorithm

        self.addAlgorithm(SolweigCalculationAlgorithm())

        # Utilities
        from .algorithms.utilities.epw_import import EpwImportAlgorithm

        self.addAlgorithm(EpwImportAlgorithm())
