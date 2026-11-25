"""Exportación del modelo promovido a ONNX (ML-06).

ML-06 T01 (ML-06-R01): `onnx_graph` construye el grafo de `logreg-1.0.0` con la biblioteca
`onnx` desde `models/logreg.json` (sin scikit-learn, sin pickle) y `package` implementa la
etapa `export`, que verifica el SHA-256 del modelo y escribe `model.onnx` en `dist/`.

Este módulo no importa `onnx`, `onnxruntime` ni scikit-learn: la CLI importa `ExportError` de
aquí sin cargarlos (se cargan solo al ejecutar `export`).
"""

from __future__ import annotations


class ExportError(RuntimeError):
    """El modelo a exportar no es el esperado (SHA-256, manifiesto o formato) o el grafo no es válido."""


__all__ = ["ExportError"]
