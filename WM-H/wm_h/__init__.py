"""WM-H instruction-first synthetic data pipeline."""

__all__ = ["InstrFirstPipeline", "InstrFirstGenerator"]


def __getattr__(name: str):
    if name == "InstrFirstPipeline":
        from .instr_first_pipeline import InstrFirstPipeline

        return InstrFirstPipeline
    if name == "InstrFirstGenerator":
        from .instr_first import InstrFirstGenerator

        return InstrFirstGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
