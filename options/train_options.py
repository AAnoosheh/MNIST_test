from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load if continuing training')

        self.parser.add_argument('--num_epochs', type=int, default=10, help='# of data epochs to run training for')
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

        self.parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for SGD')
        self.parser.add_argument('--momentum', type=float, default=0.5, help='momentum term of SGD')

        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')

        self.parser.add_argument('--save_freq', type=int, default=2, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')
