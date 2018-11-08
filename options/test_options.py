from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--which_epoch', required=True, type=int, help='which epoch to load for inference?')
