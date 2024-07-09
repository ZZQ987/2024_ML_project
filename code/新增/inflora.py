import numpy as np

from core.model.backbone.vit import Attention_LoRA

import math

import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy

from .finetune import Finetune


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

    def forward(self, x, n_type='train', get_feat=False, get_cur_feat=False):
        return self.get_logits(x, n_type, get_feat, get_cur_feat)

    def get_logits(self, x, n_type, get_feat=False, get_cur_feat=False):

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


class InfLoRA(Finetune):
    def __init__(self, backbone, feat_dim, num_class, **kwargs):

        super().__init__(backbone, feat_dim, num_class, **kwargs)

        self.kwargs = kwargs
        self.EPSILON = kwargs["EPSILON"]
        self.lamb = kwargs["lamb"]
        self.lame = kwargs["lame"]
        # task 总数
        self.task_num = kwargs["task_num"]

        self.network = Model(backbone, feat_dim, num_class, self.task_num)

        self.topk = 1
        # 目前的task
        self.cur_task = -1
        self.class_num = num_class
        self.debug = False
        # self.device =

        self.all_keys = []
        self.feature_list = []
        self.project_type = []

        self.known_classes = 0
        self.total_classes = 0

    def before_task(self, task_idx, buffer, train_loader, test_loaders):

        self.cur_task += 1
        self.network.update_fc()
        self.known_classes = self.total_classes
        self.total_classes = self.known_classes + int(self.class_num / self.task_num)

        self.network.to(self.device)

        for name, param in self.network.named_parameters():
            param.requires_grad_(False)

            if "classifier_pool" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)
            if "lora_B_k" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)
            if "lora_B_v" + "." + str(self.network.numtask - 1) in name:
                param.requires_grad_(True)

        # Double check
        # 把前面设置的，需要 require_grad_ 的，放到 enabled 里面
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
                        # 对 A 矩阵进行一个简单的初始化，后面来对A 进行训练和调整
                        module.lora_A_k[self.cur_task].weight.data.copy_(U[:, :module.rank].T / math.sqrt(3))
                        module.lora_A_v[self.cur_task].weight.data.copy_(U[:, :module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
            else:
                kk = 0
                for module in self.network.modules():
                    if isinstance(module, Attention_LoRA):
                        cur_matrix = module.cur_matrix

                        # DualGPM 的 作用，进行一定的保存
                        if self.project_type[kk] == 'remove':
                            cur_matrix = cur_matrix - torch.mm(self.feature_mat[kk], cur_matrix)
                        else:
                            assert self.project_type[kk] == 'retain'

                            cur_matrix = torch.mm(self.feature_mat[kk], cur_matrix)
                        cU, cS, cV = torch.linalg.svd(cur_matrix, full_matrices=False)
                        module.lora_A_k[self.cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                        module.lora_A_v[self.cur_task].weight.data.copy_(cU[:, :module.rank].T / math.sqrt(3))
                        module.cur_matrix.zero_()
                        module.n_cur_matrix = 0
                        kk += 1

        print(f"Parameters to be updated: {enabled}")

    def after_task(self, task_idx, buffer, train_loader, test_loaders):

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                inputs, targets = batch['image'].to(self.device), batch['label'].to(self.device)
                self.network(inputs, get_cur_feat=True)

            mat_list = []
            for module in self.network.modules():
                if isinstance(module, Attention_LoRA):
                    mat_list.append(deepcopy(module.cur_matrix))
                    module.cur_matrix.zero_()
                    module.n_cur_matrix = 0
            # self.update_GPM(mat_list)
            self.update_DualGPM(mat_list)

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

    def update_DualGPM(self, mat_list):
        threshold = (self.lame - self.lamb) * self.cur_task / self.task_num + self.lamb
        print('Threshold: ', threshold)
        if len(self.feature_list) == 0:
            # After First Task
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
                if r < (activation.shape[0] / 2):
                    self.feature_list.append(U[:, 0:max(r, 1)])
                    self.project_type.append('remove')
                else:
                    self.feature_list.append(U[:, 0:max(r, 1)])
                    self.project_type.append('retain')
        else:
            for i in range(len(mat_list)):
                if self.project_type[i] == 'remove':
                    activation = mat_list[i]
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = activation - np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()),
                                                  activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
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
                        print('Skip Updating DualGPM for layer: {}'.format(i + 1))
                        continue
                    # update GPM
                    Ui = np.hstack((self.feature_list[i], U[:, 0:r]))
                    if Ui.shape[1] > Ui.shape[0]:
                        self.feature_list[i] = Ui[:, 0:Ui.shape[0]]
                    else:
                        self.feature_list[i] = Ui
                else:
                    assert self.project_type[i] == 'retain'
                    activation = mat_list[i]
                    U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                    sval_total = (S1 ** 2).sum()
                    # Projected Representation (Eq-8)
                    act_hat = np.dot(np.dot(self.feature_list[i], self.feature_list[i].transpose()), activation)
                    U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                    # criteria (Eq-9)
                    sval_hat = (S ** 2).sum()
                    sval_ratio = (S ** 2) / sval_total
                    accumulated_sval = sval_hat / sval_total

                    r = 0
                    for ii in range(sval_ratio.shape[0]):
                        if accumulated_sval >= (1 - threshold):
                            accumulated_sval -= sval_ratio[ii]
                            r += 1
                        else:
                            break
                    if r == 0:
                        print('Skip Updating DualGPM for layer: {}'.format(i + 1))
                        continue

                    # update GPM by Projected Representation (Eq-8)
                    act_feature = self.feature_list[i] - np.dot(np.dot(U[:, 0:r], U[:, 0:r].transpose()),
                                                                self.feature_list[i])
                    Ui, Si, Vi = np.linalg.svd(act_feature)
                    self.feature_list[i] = Ui[:, :self.feature_list[i].shape[1] - r]

        print('-' * 40)
        print('Gradient Constraints Summary')
        print('-' * 40)
        for i in range(len(self.feature_list)):
            if self.project_type[i] == 'remove' and (
                    self.feature_list[i].shape[1] > (self.feature_list[i].shape[0] / 2)):
                feature = self.feature_list[i]
                # ipdb.set_trace()
                U, S, V = np.linalg.svd(feature)
                new_feature = U[:, feature.shape[1]:]
                self.feature_list[i] = new_feature
                self.project_type[i] = 'retain'
            elif self.project_type[i] == 'retain':
                assert self.feature_list[i].shape[1] <= (self.feature_list[i].shape[0] / 2)
            print('Layer {} : {}/{} type {}'.format(i + 1, self.feature_list[i].shape[1], self.feature_list[i].shape[0],
                                                    self.project_type[i]))
        print('-' * 40)
