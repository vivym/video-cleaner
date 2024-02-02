from video_cleaner.models import RAFT, RecurrentFlowCompleteNet, InpaintGenerator
from .pipeline_raft import RAFTPipeline


class PropainterPipeline:
    def __init__(
        self,
        raft_pipe: RAFTPipeline,
        flow_complete_net: RecurrentFlowCompleteNet,
        inpaint_generator: InpaintGenerator,
    ):
        self.raft_pipe = raft_pipe
        self.flow_complete_net = flow_complete_net
        self.inpaint_generator = inpaint_generator

    def __call__(self):
        ...
