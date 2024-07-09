from torch import optim

import logging
import numpy as np

from core.model.backbone.vit import Attention_LoRA

import math

import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

from .finetune import Finetune
from core.scheduler import CosineSchedule

from torch.distributions.multivariate_normal import MultivariateNormal


# 仿照 sinet
class Model(nn.Module):
    # A model consists with a backbone and a classifier
    def __init__(self, backbone, feat_dim, num_class, task_num):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.numtask = 0
        # self.classifier = nn.Linear(feat_dim, num_class)
        for module in self.backbone.modules():
            if isinstance(module, Attention_LoRA):
                module.init_param()

        self.classifier_pool = nn.ModuleList([
            nn.Linear(feat_dim, int(num_class / task_num), bias=True)
            for _ in range(task_num)
        ])

    def forward(self, x, n_type='train', get_feat=False, get_cur_feat=False, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](x)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)
        return self.get_logits(x, n_type, get_feat, get_cur_feat)

    def get_logits(self, x, n_type, get_feat=False, get_cur_feat=False):

        # 这里就是backbone
        image_features = self.backbone(x, task_id=self.numtask - 1, get_feat=get_feat,
                                       get_cur_feat=get_cur_feat)

        logits = []
        if n_type == 'train':
            for prompts in [self.classifier_pool[self.numtask - 1]]:
                logits.append(prompts(image_features))
        elif n_type == 'test':
            for prompt in self.classifier_pool[:self.numtask]:
                logits.append(prompt(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            # 'prompt_loss': prompt_loss
        }

    def update_fc(self):
        self.numtask += 1

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.backbone.feat(image, task_id=self.numtask - 1)
        else:
            image_features, _ = self.backbone.feat(image, task_id=task)
        image_features = image_features[:, 0, :]
        return image_features


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


class InfLoRA_CA1(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):

        super().__init__(backbone, feat_dim, num_class, **kwargs)

        self.kwargs = kwargs
        self.EPSILON = kwargs["EPSILON"]
        self.lamb = kwargs["lamb"]
        self.lame = kwargs["lame"]
        # task 总数
        self.task_num = kwargs["task_num"]

        self.network = Model(backbone, feat_dim, num_class, self.task_num)

        self.feature_dim = self.network.feat_dim

        self.fc_lrate = kwargs["fc_lrate"]
        self.logit_norm = 0.1
        # self.logit_norm = None

        if kwargs["dataset"] == 'cifar100':
            del self.network.backbone.feat.norm
            self.network.backbone.feat.norm = nn.LayerNorm(768)

        self.topk = 1  # origin is 5
        # self.class_num = self.network.class_num
        # 目前的task
        self.cur_task = -1
        self.class_num = num_class
        self.task_sizes = []
        self.debug = False
        # self.device =

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

        self.known_classes = 0
        self.total_classes = 0

        self.epoch = kwargs['epoch']
        self.lrate = kwargs['lrate']
        self.lrate_decay = kwargs['lrate_decay']
        self.weight_decay = kwargs['weight_decay']

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self.cur_task += 1
        self.network.update_fc()
        self.known_classes = self.total_classes
        self.total_classes = self.known_classes + int(self.class_num / self.task_num)
        self.task_sizes.append(int(self.class_num / self.task_num))

        self.network.to(self.device)

        base_params, base_fc_params = [], []
        for name, param in self.network.named_parameters():
            param.requires_grad_(False)

            if "classifier_pool" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)
                base_fc_params.append(param)
            if "lora_B_k" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)
                base_params.append(param)
            if "lora_B_v" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)
                base_params.append(param)

        # Double check
        enabled = set()
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)
                self.network(inputs, get_cur_feat=True)
                # if i > 3: break

            if self.cur_task == 0:
                for module in self.network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        U, S, V = torch.linalg.svd(cur_matrix)
                        module.lora_A_k[self.cur_task].weight.data.copy_(U[:, :module.rank].T / math.sqrt(3))
                        module.lora_A_v[self.cur_task].weight.data.copy_(U[:, :module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                kk = 0
                for module in self.network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix
                        cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk], cur_matrix)
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self.cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                        module.lora_A_v[self.cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

        print(f"Parameters to be updated: {enabled}")

    def optimizer_set(self, config):

        base_params = self.network.backbone.parameters()
        base_fc_params = [p for p in self.network.classifier_pool.parameters() if p.requires_grad == True]
        base_params = {'params': base_params, 'lr': self.lrate, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.fc_lrate, 'weight_decay': self.weight_decay}
        network_params = [base_params, base_fc_params]

        if config['optim'] == 'sgd':
            optimizer = optim.SGD(network_params, lr=self.lrate, momentum=0.9, weight_decay=self.weight_decay)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[18], gamma=self.lrate_decay)
        elif config['optim'] == 'adam':
            optimizer = optim.Adam(self.network.parameters(), lr=self.lrate, weight_decay=self.weight_decay,
                                   betas=(0.9, 0.999))
            scheduler = CosineSchedule(optimizer=optimizer, K=self.epoch)
        else:
            raise NotImplementedError

        return optimizer, scheduler

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)
                self.network(inputs, get_cur_feat=True)
                # if i > 3: break

            mat_list = []
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            self.update_GPM(mat_list)

            # Projection Matrix Precomputation
            self.feature_mat = []
            for p in range(len(self.feature_list)):
                Uf = torch.Tensor(np.dot(self.feature_list[p], self.feature_list[p].transpose()))
                print('Layer {} - Projection Matrix shape: {}'.format(p + 1, Uf.shape))
                self.feature_mat.append(Uf)

    def observe(self, data):

        # 改了数据集读取，没改backbone
        x, y = data['image'], data['label']
        x, y = x.to(self.device), y.to(self.device)
        mask = (y >= self.known_classes).nonzero().view(-1)
        x = torch.index_select(x, 0, mask)
        y = torch.index_select(y, 0, mask) - self.known_classes

        logits = self.network(x)['logits']
        loss = F.cross_entropy(logits, y)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == y).item()

        correct = pred.eq(y.expand_as(pred)).cpu().sum()

        total = len(y)
        a = correct / total
        b = acc / x.size(0)

        return pred, acc / x.size(0), loss

    def inference(self, data):

        x, y = data['image'], data['label']
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            logits = self.network(x, 'test')['logits']

        predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(predicts == y).item()

        correct = pred.eq(y.expand_as(pred)).cpu().sum()
        total = len(y)

        a = correct / total
        b = acc / x.size(0)

        return pred, acc / x.size(0)

    def get_parameters(self, config):
        return self.network.parameters()

    def stage_2(self, stage2_dataloader, stage2_test_loaders):

        print("================ InfLoRA_CA1 (stage2)================")
        self._compute_class_mean(stage2_dataloader, check_diff=False, oracle=False)
        if self.cur_task > 0:
            self._stage2_compact_classifier(self.task_sizes[-1], stage2_test_loaders)

    def update_GPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self.cur_task / self.task_num + self.lamb
        print('Threshold: ', threshold)
        if len(self.feature_list) == 0:
            # After First Task
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                # U=self.pq(torch.from_numpy(U), 0.25)
                # U = U.numpy()
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
                self.feature_list.append(U[:, 0:max(r, 1)])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()),
                                              activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # U=self.pq(torch.from_numpy(U), 0.25)
                # U = U.numpy()
                # criteria (Eq-9)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating GPM for layer: {}'.format(i + 1))
                    continue
                # update GPM
                Ui = np.hstack((self.feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    self.feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    self.feature_list[i] = Ui

        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            print('Layer {} : {}/{}'.format(i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0]))
        print('-' * 40)

    def _stage2_compact_classifier(self, task_size, stage2_test_loaders):
        for p in self.network.classifier_pool[:self.cur_task + 1].parameters():
            p.requires_grad = True

        run_epochs = 5
        crct_num = self.total_classes
        param_list = [p for p in self.network.classifier_pool.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': 0.01,
                           'weight_decay': 0.0005}]
        optimizer = optim.SGD(network_params, lr=0.01, momentum=0.9, weight_decay=0.0005)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self.network.to(self.device)

        self.network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256

            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self.cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self.device) * (
                        0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self.device)
                cls_cov = self._class_covs[c_id].to(self.device)

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

            sampled_data = torch.cat(sampled_data, dim=0).float().to(self.device)
            sampled_label = torch.tensor(sampled_label).long().to(self.device)

            inputs = sampled_data
            targets = sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            for _iter in range(crct_num):
                inp = inputs[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                tgt = targets[_iter * num_sampled_pcls:(_iter + 1) * num_sampled_pcls]
                outputs = self.network(inp, fc_only=True)
                logits = outputs

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self.cur_task + 1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)

                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self.network, stage2_test_loaders)
            # stage2_test_loader 其实是多个test_loader 的组合
            # 原论文比如说，不同的task test的之后，是8,16,24组合
            # libcon 中是 8 （8，8） （8，8，8） 组合
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self.cur_task, losses / self.total_classes, test_acc)
            logging.info(info)

    def _compute_accuracy(self, model, loaders):
        model.eval()
        correct, total = 0, 0
        for t, loader in enumerate(loaders):
            for batch_idx, batch in enumerate(loader):
                x, y = batch['image'], batch['label']
                x, y = x.to(self.device), y.to(self.device)

                with torch.no_grad():
                    logits = self.network(x, 'test')['logits']

                predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1].view(-1)  # [bs, topk]

                pred = torch.argmax(logits, dim=1)
                acc = torch.sum(predicts == y).item()

                correct += pred.eq(y.expand_as(pred)).cpu().sum()
                total += len(y)

        return correct / total

    def _compute_class_mean(self, stage2_dataloader, check_diff=False, oracle=False):
        if hasattr(self, '_class_means') and self._class_means is not None and not check_diff:
            ori_classes = self._class_means.shape[0]
            assert ori_classes == self.known_classes
            new_class_means = np.zeros((self.total_classes, self.feature_dim))
            new_class_means[:self.known_classes] = self._class_means
            self._class_means = new_class_means
            # new_class_cov = np.zeros((self.total_classes, self.feature_dim, self.feature_dim))
            new_class_cov = torch.zeros((self.total_classes, self.feature_dim, self.feature_dim))
            new_class_cov[:self.known_classes] = self._class_covs
            self._class_covs = new_class_cov
        elif not check_diff:
            self._class_means = np.zeros((self.total_classes, self.feature_dim))
            # self._class_covs = np.zeros((self.total_classes, self.feature_dim, self.feature_dim))
            self._class_covs = torch.zeros((self.total_classes, self.feature_dim, self.feature_dim))

        for class_idx in range(self.known_classes, self.total_classes):

            idx_loader = stage2_dataloader[class_idx - self.known_classes]
            vectors, _ = self._extract_vectors(idx_loader)

            class_mean = np.mean(vectors, axis=0)
            # class_cov = np.cov(vectors.T)
            class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T) + torch.eye(class_mean.shape[-1]) * 1e-4
            if check_diff:
                log_info = "cls {} sim: {}".format(class_idx, torch.cosine_similarity(
                    torch.tensor(self._class_means[class_idx, :]).unsqueeze(0),
                    torch.tensor(class_mean).unsqueeze(0)).item())
                logging.info(log_info)
                np.save('task_{}_cls_{}_mean.npy'.format(self.cur_task, class_idx), class_mean)
                np.save('task_{}_cls_{}_mean_beforetrain.npy'.format(self.cur_task, class_idx),
                        self._class_means[class_idx, :])
                # print(class_idx, torch.cosine_similarity(torch.tensor(self._class_means[class_idx, :]).unsqueeze(0), torch.tensor(class_mean).unsqueeze(0)))
            self._class_means[class_idx, :] = class_mean
            self._class_covs[class_idx, ...] = class_cov
            # self._class_covs.append(class_cov)

    def _extract_vectors(self, loader):

        self.network.eval()
        vectors, targets = [], []

        for batch_idx, batch in enumerate(loader):
            _inputs, _targets = batch['image'].to(self.device), batch['label'].to(self.device)

            _targets = _targets.cpu().numpy()

            _vectors = tensor2numpy(self.network.extract_vector(_inputs.to(self.device)))

            vectors.append(_vectors)
            targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
