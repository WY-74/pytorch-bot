from utils.base import Timer
from utils.base import try_all_gpus


def train(net, train_iter, test_iter, loss, optimer, num_epochs, devices=try_all_gpus(), data_parallel: bool = True):
    num_batches = len(train_iter)
    if not devices:
        devices = ["cpu"]

    timer = Timer()
    animator = Animator(
        xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc']
    )

    net = torch.jit.script(net)
    if devices and len(devices) > 1:
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    elif devices and len(devices) == 1:
        net = net.to(devices[0])
    else:
        pass

    for epoch in range(num_epochs):
        metric = Accumulator(4)
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            if isinstance(X, list):
                X = [x.to(devices[0]) for x in X]
            else:
                X = X.to(devices[0])
            y = y.to(devices[0])

            net.train()
            optimer.zero_grad()
            pred = net(X)
            l = loss(pred, y)
            l.sum().backward()
            optimer.step()
            train_loss_sum = l.sum()  # 总损失
            train_acc_sum = metric.accuracy(pred, y)  # 预测正确个数

            timer.stop()
            metric.add(
                train_loss_sum, train_acc_sum, y.shape[0], y.numel()
            )  # 总损失, 预测正确个数, 训练图片总数, 总像素个数
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (metric[0] / metric[2], metric[1] / metric[3], None))
        test_acc = _evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    print(f'loss {metric[0] / metric[2]:.3f}, train acc ' f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on ' f'{str(devices)}')
