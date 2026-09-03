# Third-party lineage

The BOTA experiments use a T5 text-to-text recommendation setup derived from
the public [E2URec](https://github.com/justarter/E2URec) research pipeline and
include an explicitly adapted E2URec teacher/student baseline for the frozen
BOTA requests and LoRA coordinate. The adaptation is identified in code as
`E2URec-Short-FixedAB`; it is not presented as an unchanged execution of the
upstream repository.

Dataset and pretrained-model files are not redistributed here. Users are
responsible for observing the licenses and terms of MovieLens, GoodReads, T5,
E2URec and any separately downloaded baseline artifacts.

This notice is attribution and scope documentation; it does not replace the license terms of any upstream project.
