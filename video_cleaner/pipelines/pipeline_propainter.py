from video_cleaner.models import RAFT, RecurrentFlowCompleteNet, InpaintGenerator


class PropainterPipeline:
    def __init__(
        self,
        raft: RAFT,
        flow_complete_net: RecurrentFlowCompleteNet,
        inpaint_generator: InpaintGenerator,
    ):
        ...

    def __call__(self):
        ...
