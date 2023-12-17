from rabetorch.builders.pipeline_builder import PipelineBuilder
from rabetorch.builders.model_builder import ModelBuilder
from rabetorch.builders.solver_builder import SolverBuilder
from rabetorch.builders.loss_builder import LossBuilder


class TrainingRunner():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def run_train(self) -> None:
        """Running script of train.
        
            1. build pipeline:
                Pipeline can be constructed by multiple datasets.
            2. build model:
                Model need to have backbone and head. Neck is optional.
            3. build solver:
                Set optimizer.
            4. build loss:
                Set loss.
            5. Training
            6. Evaluating
        """
        # build pipeline
        train_pipeline = PipelineBuilder(self.cfg.DATA, is_train=True)
        test_pipeline = PipelineBuilder(self.cfg.DATA, is_train=False)

        train_loader = train_pipeline.build_pipeline()
        test_loader = test_pipeline.build_pipeline()

        # build model
        model_builder = ModelBuilder(self.cfg.MODEL)
        model = model_builder.build_model()
        print("Model Summary")
        print(model)

        # build solver
        solver_builder = SolverBuilder(self.cfg.SOLVER)
        optimizer, max_epoch = solver_builder.build_solver(model)

        # build loss
        loss_builder = LossBuilder(self.cfg.SOLVER)
        criterion = loss_builder.build_loss()

        # run train
        print("Start train from scratch. Max epoch: {}".format(max_epoch))
        for epoch in range(max_epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            print('Epoch: {}/{}, Loss: {}'.format(epoch + 1, max_epoch, loss.item()))
        print("train done")

        # run evaluation
        print("start evaluation")
        model.eval()
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            output = model(data)
            pred = output.argmax(dim=1)
            print(target, pred)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        print('Accuracy: {}'.format(100 * correct / total))
