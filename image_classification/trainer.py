import os
from torch.utils import data
from torch.utils.data import DataLoader

from utils import torch, nn
from utils import base, Timer, Animator


class Trainer:
    def _get_devices(self, num_gpus):
        devices = base.try_all_gpus()
        if not devices:
            print("[Trainer][_get_devices] devices: cpu")
            print("[Trainer][_get_devices] The num_gpus is invalid")
            return torch.device("cpu")

        if num_gpus > len(devices):
            raise Exception(f"The num_gpus exceeds the total number of GPUs({len(devices)}).")

        print(f"[Trainer][_get_devices] devices: {devices}.")
        return devices[:num_gpus]

    def _init_net(self, net, devices):
        net = net.apply(self.init_weight)
        net = torch.jit.script(net)
        if isinstance(devices, list):
            net = net.to(devices[0])
            net = nn.DataParallel(net, device_ids=devices)

        print(f"[Trainer][_init_net] Init net finished.")
        return net

    def _init_dataiter(self, raw_data, shuffle=True, *, batch_size=256, num_workers=None):
        if raw_data is None:
            print("[Trainer][_init_dataiter] The dataiter is None.")
            return None

        num_workers = self._get_num_workers() if not num_workers else num_workers 
        if not isinstance(raw_data, DataLoader):
            print(f"[Trainer][_init_dataiter] batch_size: {batch_size}; num_workers: {num_workers}")
            dataset = data.TensorDataset(*raw_data)
            data_iter = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            print("[Trainer][_init_dataiter] the data is already of type DataLoader.")
            return raw_data

        return data_iter

    def _get_num_workers(self):
        num = os.cpu_count()
        return num

    def init_weight(self, m):
        pass

    def loss(self):
        """ return nn.CrossEntropyLoss() """
        pass

    def optimizer(self, net, lr):
        """ return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=5e-4) """
        pass

    def run(
        self, 
        net,
        train_iter, test_iter,
        *,
        num_gpus = None, num_workers = None,
        batch_size=256, num_epochs=10, lr=1e-3,
    ):
        """
        执行训练:
            - net: 深度学习网络.
            - train_iter: 训练集.
            - test_iter: 验证集, 可以为None.
            - num_gpus: 使用gpu个数, 默认为None.
            - num_workers: 读取数据时使用的子进程数, 默认为None. 若设置为None则会使用os.cpu_count()结果作为子进程数.
            - batch_size: 批量大小, 默认为256.
            - num_epochs: 训练总轮数, 默认为10.
            - lr: 学习率, 默认为1e-3.
        """
        # 初始化
        devices = self._get_devices(num_gpus)
        net = self._init_net(net, devices)
        
        if train_iter is None:
            raise Exception("The train_iter is None")
        train_iter = self._init_dataiter(train_iter, batch_size=batch_size, num_workers=num_workers)
        test_iter = self._init_dataiter(test_iter, shuffle=False, batch_size=batch_size, num_workers=num_workers)
        loss = self.loss()
        optimizer = self.optimizer(net, lr)

        # 开始训练
        print("[Trainer][run] Start training")
        timer = Timer()

        if test_iter:
            animator = Animator(xlabel='epoch', xlim=(1, num_epochs), legend=['train loss', 'train acc', 'test acc'])
        else:
            animator = Animator(xlabel='epoch', xlim=(1, num_epochs), legend=['train loss', 'train acc'])
            
        num_batches = len(train_iter)    # 总批次数
        print(f"[Trainer][run] num_epochs: {num_epochs}; num_batches: {num_batches}")
        for epoch in range(num_epochs):
            metric = base.Accumulator(3)
            net.train()
            timer.start()
            for i, (X, y) in enumerate(train_iter):
                if isinstance(devices, list):
                    X, y = X.to(devices[0], non_blocking=True), y.to(devices[0], non_blocking=True)

                optimizer.zero_grad()
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                
                with torch.no_grad():
                    metric.add(l.detach() * X.shape[0], metric.accuracy(y_hat, y), X.shape[0])  # 单批次总损失，正确个数，总数
                timer.stop()

                train_l = metric[0] / metric[2]  # 随着小批量数据扫描，当前的平均损失
                train_acc = metric[1] / metric[2]  # 随着小批量数据扫描，当前的准确率

                if ((i + 1) % (num_batches // 5)) == 0 or (num_batches - 1 == i):
                    # 每num_batches // 5个批次打印一次
                    animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))

            if test_iter:
                test_acc = base.evaluate_accuracy_gpu(net, test_iter, device=devices[0])
                animator.add(epoch + 1, (None, None, test_acc))
            else:
                test_acc = 0

        print(f"loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}")
        print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}")
        return train_acc, test_acc

    