import argparse

from omegaconf import OmegaConf

from rabetorch.util.config import load_config, smart_type
from rabetorch.builders.pipeline_builder import PipelineBuilder
from rabetorch.builders.model_builder import ModelBuilder
from rabetorch.builders.solver_builder import SolverBuilder
from rabetorch.builders.loss_builder import LossBuilder


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "config",
        type=str,
        help="Relative path to main config file from ./configs/"
    )
    p.add_argument("override", nargs="*", type=smart_type, help="Override config")

    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    override_dict = None
    if args.override:
        it = iter(args.override)
        override_dict = dict(zip(it, it))
    config_path = "./configs/" + args.config + ".yaml"
    cfg = load_config(config_path, override_dict)

    # build pipeline
    train_pipeline = PipelineBuilder(cfg.DATA, is_train=True)
    test_pipeline = PipelineBuilder(cfg.DATA, is_train=False)

    train_loader = train_pipeline.build_pipeline()
    test_loader = test_pipeline.build_pipeline()

    # build model
    model_builder = ModelBuilder(cfg.MODEL)
    model = model_builder.build_model()
    print("Model Summary")
    print(model)

    # build solver
    solver_builder = SolverBuilder(cfg.SOLVER)
    optimizer, max_epoch = solver_builder.build_solver(model)

    # build loss
    loss_builder = LossBuilder(cfg.SOLVER)
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
