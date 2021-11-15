from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, model):
        self.model = model
        self.step = 0
        self.writer = SummaryWriter()

    def on_step(self):
        self.step += 1

    def report_output(self, actual, predicted):
        self.writer.add_histogram(
            'output/predicted',
            predicted,
            global_step=self.step
        )
        self.writer.add_histogram(
            'output/error',
            predicted - actual,
            global_step=self.step
        )

    def report_train_loss(self, train):
        self.writer.add_scalar('loss/train', train, global_step=self.step)

    def report_validation_loss(self, validation):
        self.writer.add_scalar('loss/validation', validation, global_step=self.step)

    def report_model_parameters(self):
        for name, parameter in self.model.named_parameters():
            key = 'parameters/' + name.replace('.', '/')
            self.writer.add_histogram(
                key,
                parameter.data,
                global_step=self.step
            )
