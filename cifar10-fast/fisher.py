import torch
import torch.nn as nn
from su_torch_backend import DataLoader
from itertools import chain
import math
import numpy as np


CLASSIFIER_NAME = "fc"


def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    criterion = nn.CrossEntropyLoss()
    
    idx = 0

    # data_list = []

    # class_num = np.array([0 for _ in range(10)])

    # max_num_per_class = math.ceil(num_samples / 10)

    # num_allow_max = num_samples - 10 * int(num_samples / 10)

    # for inputs in data_loader:
    #     target = inputs["target"].item()

    #     if class_num[target] >= max_num_per_class:
    #         continue
        
    #     num_max = (class_num == max_num_per_class).sum()
    #     if num_max == num_allow_max:
    #         # cannot allow more to reach max size
    #         if class_num[target] + 1 == max_num_per_class:
    #             continue

    #     class_num[target] += 1

    #     data_list.append(inputs)

    #     if np.sum(class_num) >= num_samples:
    #         break

    for inputs in data_loader:
        if idx >= num_samples:
            break

        # print(idx)
        logits = model(inputs)["logits"]
        labels = inputs["target"]

        loss = criterion(logits, labels)

        loss.backward()

        for name, param in model.named_parameters():
            gradients_dict[name] += grad_method(param.grad).data

        model.zero_grad()

        idx += 1

    return gradients_dict


def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, (inputs, labels) in enumerate(data_loader):
        if idx >= num_samples:
            break

        inputs = inputs.to(cuda_device)
        labels = labels.to(cuda_device)

        logits = model(inputs)

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict


class FisherMask:
    def __init__(self, model, train_dataset, num_samples, keep_ratio, sample_type, grad_type):
        self.model = model
        self.train_dataset = train_dataset
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.sample_type = sample_type
        self.grad_type = grad_type

        self.mask = None

    def calculate_mask(self):
        model = self.model
        train_dataset = self.train_dataset
        num_samples = self.num_samples
        keep_ratio = self.keep_ratio
        sample_type = self.sample_type
        grad_type = self.grad_type

        original_device = list(model.parameters())[0].device
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(cuda_device)

        data_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=1, set_random_choices=True)

        if sample_type == "label":
            importance_method = calculate_the_importance_label
        elif sample_type == "expect":
            importance_method = calculate_the_importance_expect
        else:
            raise NotImplementedError

        gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

        # add sizes and aggregate tensors
        sizes = {}
        tensors = []

        classifier_size = 0
        all_params_size = 0

        classifier_mask_dict = {}

        for k, v in gradients.items():
            # don't count classifier layer, they should be all trainable
            if CLASSIFIER_NAME in k:
                classifier_size += torch.prod(torch.tensor(v.shape)).item()
                classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
            else:
                sizes[k] = v.shape
                tensors.append(v.view(-1))

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        tensors = torch.cat(tensors, 0)

        keep_num = int(all_params_size * keep_ratio) - classifier_size

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]

        masks = torch.zeros_like(tensors, device=cuda_device)

        masks[top_pos] = 1

        assert masks.long().sum() == len(top_pos)

        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
            now_idx = end_idx

        assert now_idx == len(masks)

        # Add the classifier's mask to mask_dict
        mask_dict.update(classifier_mask_dict)

        model.to(original_device)

        # Print the parameters for checking
        classifier_size = 0
        all_params_size = 0
        pretrain_weight_size = 0
        
        for k, v in mask_dict.items():
            if CLASSIFIER_NAME in k:
                classifier_size += (v == 1).sum().item()
            else:
                pretrain_weight_size += (v == 1).sum().item()

            all_params_size += torch.prod(torch.tensor(v.shape)).item()
        
        print(pretrain_weight_size, classifier_size, all_params_size)
        print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

        self.mask = mask_dict.values()


class RandomFisherMask:
    def __init__(self, model, train_dataset, num_samples, keep_ratio, sample_type, grad_type):
        self.model = model
        self.train_dataset = train_dataset
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.sample_type = sample_type
        self.grad_type = grad_type

        self.mask = None

    def calculate_mask(self):
        model = self.model
        train_dataset = self.train_dataset
        num_samples = self.num_samples
        keep_ratio = self.keep_ratio
        sample_type = self.sample_type
        grad_type = self.grad_type

        original_device = list(model.parameters())[0].device
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(cuda_device)

        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = torch.rand(param.shape).to(original_device)

        # add sizes and aggregate tensors
        sizes = {}
        tensors = []

        classifier_size = 0
        all_params_size = 0

        classifier_mask_dict = {}

        for k, v in gradients.items():
            # don't count classifier layer, they should be all trainable
            if CLASSIFIER_NAME in k:
                classifier_size += torch.prod(torch.tensor(v.shape)).item()
                classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
            else:
                sizes[k] = v.shape
                tensors.append(v.view(-1))

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        tensors = torch.cat(tensors, 0)

        keep_num = int(all_params_size * keep_ratio) - classifier_size

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]

        masks = torch.zeros_like(tensors, device=original_device)

        masks[top_pos] = 1

        assert masks.long().sum() == len(top_pos)

        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
            now_idx = end_idx

        assert now_idx == len(masks)

        # Add the classifier's mask to mask_dict
        mask_dict.update(classifier_mask_dict)

        model.to(original_device)

        # Print the parameters for checking
        classifier_size = 0
        all_params_size = 0
        pretrain_weight_size = 0
        
        for k, v in mask_dict.items():
            if CLASSIFIER_NAME in k:
                classifier_size += (v == 1).sum().item()
            else:
                pretrain_weight_size += (v == 1).sum().item()

            all_params_size += torch.prod(torch.tensor(v.shape)).item()
        
        print(pretrain_weight_size, classifier_size, all_params_size)
        print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")
        
        self.mask = mask_dict.values()


class FullOneFisherMask:
    def __init__(self, model, *args, **kwargs):
        self.model = model
        self.mask = None

    def calculate_mask(self):
        model = self.model

        mask = {}
        
        for n, p in model.named_parameters():
            mask[n] = torch.ones_like(p)

        self.mask = mask.values()


class MultiFisherMask:
    def __init__(self, model, train_dataset, num_samples, keep_ratio, sample_type, grad_type, split):
        self.model = model
        self.train_dataset = train_dataset
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.sample_type = sample_type
        self.grad_type = grad_type
        self.split = split

    def calculate_mask(self):
        model = self.model
        train_dataset = self.train_dataset
        num_samples = self.num_samples
        keep_ratio = self.keep_ratio
        sample_type = self.sample_type
        grad_type = self.grad_type
        split = self.split

        original_device = list(model.parameters())[0].device
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(cuda_device)

        data_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=1, set_random_choices=True)

        if sample_type == "label":
            importance_method = calculate_the_importance_label
        elif sample_type == "expect":
            importance_method = calculate_the_importance_expect
        else:
            raise NotImplementedError

        gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

        # add sizes and aggregate tensors
        sizes = {}
        tensors = []

        classifier_size = 0
        all_params_size = 0

        classifier_mask_dict = {}

        for k, v in gradients.items():
            # don't count classifier layer, they should be all trainable
            if CLASSIFIER_NAME in k:
                classifier_size += torch.prod(torch.tensor(v.shape)).item()
                classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
            else:
                sizes[k] = v.shape
                tensors.append(v.view(-1))

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        tensors = torch.cat(tensors, 0)

        keep_num = int(all_params_size * keep_ratio) - classifier_size

        keep_num = split * keep_num

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]

        # random_indices = torch.randperm(len(top_pos), device=cuda_device)
        
        # Actually is interleave, just not change the naming used before
        random_indices = [torch.arange(i, len(top_pos), split) for i in range(split)]
        random_indices = torch.hstack(random_indices)

        random_indices = random_indices.reshape(split, -1)

        mask_list = []

        for i in range(random_indices.shape[0]):
            # get the indices of the the split
            split_indices = random_indices[i]
            split_top_pos = top_pos[split_indices]
            
            masks = torch.zeros_like(tensors, device=cuda_device)

            masks[split_top_pos] = 1

            assert masks.sum() == len(split_top_pos)

            mask_dict = {}

            now_idx = 0
            for k, v in sizes.items():
                end_idx = now_idx + torch.prod(torch.tensor(v))
                mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
                now_idx = end_idx

            assert now_idx == len(masks)

            # Add the classifier's mask to mask_dict
            mask_dict.update(classifier_mask_dict)

            model.to(original_device)

            # Print the parameters for checking
            classifier_size = 0
            all_params_size = 0
            pretrain_weight_size = 0
            
            for k, v in mask_dict.items():
                if CLASSIFIER_NAME in k:
                    classifier_size += (v == 1).sum().item()
                else:
                    pretrain_weight_size += (v == 1).sum().item()

                all_params_size += torch.prod(torch.tensor(v.shape)).item()
            
            print(pretrain_weight_size, classifier_size, all_params_size)
            print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

            mask_list.append(mask_dict.values())

        return mask_list


class MultiSameFisherMask:
    def __init__(self, model, train_dataset, num_samples, keep_ratio, sample_type, grad_type, split):
        self.model = model
        self.train_dataset = train_dataset
        self.num_samples = num_samples
        self.keep_ratio = keep_ratio
        self.sample_type = sample_type
        self.grad_type = grad_type
        self.split = split

    def calculate_mask(self):
        model = self.model
        train_dataset = self.train_dataset
        num_samples = self.num_samples
        keep_ratio = self.keep_ratio
        sample_type = self.sample_type
        grad_type = self.grad_type
        split = self.split

        original_device = list(model.parameters())[0].device
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(cuda_device)

        data_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=1, set_random_choices=True)

        if sample_type == "label":
            importance_method = calculate_the_importance_label
        elif sample_type == "expect":
            importance_method = calculate_the_importance_expect
        else:
            raise NotImplementedError

        gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

        # add sizes and aggregate tensors
        sizes = {}
        tensors = []

        classifier_size = 0
        all_params_size = 0

        classifier_mask_dict = {}

        for k, v in gradients.items():
            # don't count classifier layer, they should be all trainable
            if CLASSIFIER_NAME in k:
                classifier_size += torch.prod(torch.tensor(v.shape)).item()
                classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
            else:
                sizes[k] = v.shape
                tensors.append(v.view(-1))

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        tensors = torch.cat(tensors, 0)

        keep_num = int(all_params_size * keep_ratio) - classifier_size

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]
            
        masks = torch.zeros_like(tensors, device=cuda_device)

        masks[top_pos] = 1

        assert masks.long().sum() == len(top_pos)

        mask_dict = {}

        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
            now_idx = end_idx

        assert now_idx == len(masks)

        # Add the classifier's mask to mask_dict
        mask_dict.update(classifier_mask_dict)

        model.to(original_device)

        # Print the parameters for checking
        classifier_size = 0
        all_params_size = 0
        pretrain_weight_size = 0
        
        for k, v in mask_dict.items():
            if CLASSIFIER_NAME in k:
                classifier_size += (v == 1).sum().item()
            else:
                pretrain_weight_size += (v == 1).sum().item()

            all_params_size += torch.prod(torch.tensor(v.shape)).item()
        
        print(pretrain_weight_size, classifier_size, all_params_size)
        print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

        mask_list = [mask_dict.values() for _ in range(split)]
        
        return mask_list


class MultiFullOneFisherMask:
    def __init__(self, model, *args, split=2, **kwargs):
        self.model = model
        self.split = split

    def calculate_mask(self):
        model = self.model
        split = self.split

        mask_list = []
        for _ in range(split):
            mask = {}
            
            for n, p in model.named_parameters():
                mask[n] = torch.ones_like(p)

            mask_list.append(mask.values())

        return mask_list


class MultiRandomFisherMask:
    def __init__(self, model, train_dataset, num_samples, keep_ratio, sample_type, grad_type, split):
        self.model = model
        self.split = split
        self.keep_ratio = keep_ratio

    def calculate_mask(self):
        model = self.model
        split = self.split
        keep_ratio = self.keep_ratio

        original_device = list(model.parameters())[0].device
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(cuda_device)

        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = torch.rand(param.shape).to(original_device)

        # add sizes and aggregate tensors
        sizes = {}
        tensors = []

        classifier_size = 0
        all_params_size = 0

        classifier_mask_dict = {}

        for k, v in gradients.items():
            # don't count classifier layer, they should be all trainable
            if CLASSIFIER_NAME in k:
                classifier_size += torch.prod(torch.tensor(v.shape)).item()
                classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
            else:
                sizes[k] = v.shape
                tensors.append(v.view(-1))

            all_params_size += torch.prod(torch.tensor(v.shape)).item()

        tensors = torch.cat(tensors, 0)

        keep_num = int(all_params_size * keep_ratio) - classifier_size

        keep_num = split * keep_num

        assert keep_num > 0

        top_pos = torch.topk(tensors, keep_num)[1]

        # random_indices = torch.randperm(len(top_pos), device=cuda_device)
        
        # Actually is interleave, just not change the naming used before
        random_indices = [torch.arange(i, len(top_pos), split) for i in range(split)]
        random_indices = torch.hstack(random_indices)

        random_indices = random_indices.reshape(split, -1)

        mask_list = []

        for i in range(random_indices.shape[0]):
            # get the indices of the the split
            split_indices = random_indices[i]
            split_top_pos = top_pos[split_indices]
            
            masks = torch.zeros_like(tensors, device=cuda_device)

            masks[split_top_pos] = 1

            assert masks.sum() == len(split_top_pos)

            mask_dict = {}

            now_idx = 0
            for k, v in sizes.items():
                end_idx = now_idx + torch.prod(torch.tensor(v))
                mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
                now_idx = end_idx

            assert now_idx == len(masks)

            # Add the classifier's mask to mask_dict
            mask_dict.update(classifier_mask_dict)

            model.to(original_device)

            # Print the parameters for checking
            classifier_size = 0
            all_params_size = 0
            pretrain_weight_size = 0
            
            for k, v in mask_dict.items():
                if CLASSIFIER_NAME in k:
                    classifier_size += (v == 1).sum().item()
                else:
                    pretrain_weight_size += (v == 1).sum().item()

                all_params_size += torch.prod(torch.tensor(v.shape)).item()
            
            print(pretrain_weight_size, classifier_size, all_params_size)
            print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")

            mask_list.append(mask_dict.values())

        return mask_list