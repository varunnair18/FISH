import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from time import time

from backpack import backpack, extend
from backpack.extensions import (
    GGNMP,
    HMP,
    KFAC,
    KFLR,
    KFRA,
    PCHMP,
    BatchGrad,
    BatchL2Grad,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SumGradSquared,
    Variance,
)

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertConfig, AutoModelForSequenceClassification

from transformers.models.bert.modeling_bert import (
    BertPooler,
    BertLayer,
    BertOutput,
    BertEmbeddings
)


cuda_device = "cpu"

def compute_time(func):
    def wrap(*args, **kwargs):
        s = time()
        out = func(*args, **kwargs)

        print(f"spent: {time() - s: .4f} s")

        return out

    return wrap

@compute_time
def compute_fisher(model, dset, num_samples):
    data_loader = DataLoader(dset, batch_size=1)
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    grad_method = torch.square

    a = 0
    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break
        inputs = torch.cat(inputs, dim=0).to(cuda_device)
        logits = model(inputs)["logits"]

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data


                    if name == "bert_embeddings.position_embeddings.weight":
                        a += param.grad

                model.zero_grad()


    print(a)

    return gradients_dict


@compute_time
def compute_fisher_inferonce(model, dset, num_samples, batch_size):
    data_loader = DataLoader(dset, batch_size=batch_size)
    gradients_dict = {}
    
    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    grad_method = torch.square

    accumm_data = 0

    for idx, inputs in enumerate(data_loader):
        if accumm_data >= num_samples:
            break
        print(accumm_data)
        inputs = torch.cat(inputs, dim=0).to(cuda_device)
        accumm_data += inputs.shape[0]

        logits = model(inputs)["logits"]

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


@compute_time
def compute_fisher_batchwise(model, dset, num_samples, batch_size, num_classes):
    data_loader = DataLoader(dset, batch_size=batch_size)

    loss_function = nn.CrossEntropyLoss(reduction="sum")
    model = extend(model)

    loss_function = extend(loss_function)

    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    grad_method = torch.square

    accumm_data = 0
    for idx, inputs in enumerate(data_loader):
        if accumm_data >= num_samples:
            break

        inputs = torch.cat(inputs, dim=0).to(cuda_device)
        accumm_data += inputs.shape[0]
        
        N = inputs.shape[0]

        if len(inputs.shape) == 2:
            inputs = inputs.repeat(num_classes, 1)
        elif len(inputs.shape) == 3:
            inputs = inputs.repeat(num_classes, 1, 1)
        else:
            raise NotImplementedError

        logits = model(inputs)["logits"]

        y = torch.arange(num_classes)

        y = torch.repeat_interleave(y, N, dim=0).to(cuda_device)

        probs = torch.nn.functional.softmax(logits, -1)

        loss = loss_function(logits, y)

        with backpack(BatchGrad()):
            loss.backward()

        prob = torch.gather(probs, dim=-1, index=y.unsqueeze(-1))

        for name, param in model.named_parameters():
            size = [1 for _ in range(len(param.shape))]
            size.insert(0, prob.shape[0])
            prob = prob.reshape(size)

            gradients_dict[name] += (prob * grad_method(param.grad_batch)).data.sum(0)


        model.zero_grad()

    return gradients_dict


@compute_time
def compute_fisher_opacus(model, dset, num_samples, batch_size, num_classes):
    import opacus
    data_loader = DataLoader(dset, batch_size=batch_size)

    loss_function = nn.CrossEntropyLoss(reduction="sum")

    opacus.autograd_grad_sample.add_hooks(model, loss_reduction="sum")

    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    grad_method = torch.square

    accumm_data = 0
    for idx, inputs in enumerate(data_loader):
        if accumm_data >= num_samples:
            break

        inputs = torch.cat(inputs, dim=0).to(cuda_device)
        accumm_data += inputs.shape[0]
        
        N = inputs.shape[0]

        if len(inputs.shape) == 2:
            inputs = inputs.repeat(num_classes, 1)
        elif len(inputs.shape) == 3:
            inputs = inputs.repeat(num_classes, 1, 1)
        else:
            raise NotImplementedError

        logits = model(inputs)["logits"]

        y = torch.arange(num_classes)

        y = torch.repeat_interleave(y, N, dim=0).to(cuda_device)

        probs = torch.nn.functional.softmax(logits, -1)

        loss = loss_function(logits, y)

        with backpack(BatchGrad()):
            loss.backward()

        prob = torch.gather(probs, dim=-1, index=y.unsqueeze(-1))

        for name, param in model.named_parameters():
            size = [1 for _ in range(len(param.shape))]
            size.insert(0, prob.shape[0])
            prob = prob.reshape(size)

            gradients_dict[name] += (prob * grad_method(param.grad_sample)).data.sum(0)

            del param.grad_sample


        model.zero_grad()

    return gradients_dict


class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_1 = nn.Linear(dim, 20)
        self.layer_2 = nn.Linear(20, 2)

    def forward(self, inputs):
        return self.layer_2(nn.functional.relu(self.layer_1(inputs)))


class ResLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

        # self.norm.weight.data.normal_(0, 1)
        # self.norm.bias.data.normal_(0, 1)

    def forward(self, inputs_1, inputs_2, divide_2=True):
        inputs_1 = self.linear(inputs_1)
        inputs_1 = inputs_1
        out = self.norm(inputs_1 + inputs_2)
        if divide_2:
            out = out / 2
        return SequenceClassifierOutput(logits=out)


class ReturnList(nn.Module):
    def __init__(self, dim_1, dim_2, dim_out):
        super().__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(dim_1, dim_out),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(dim_2, dim_out),
            nn.ReLU()
        )

    def forward(self, inputs_1, inputs_2):
        out1 = self.linear1(inputs_1)
        out2 = self.linear2(inputs_2)

        return out1 + out2, out1 - out2


class ResModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_1 = nn.Linear(dim, 20)
        self.layer_2 = nn.Linear(20, 2)
        self.layer_3 = nn.Linear(20, 2)

        self.listlayer = ReturnList(2, 2, 2)

        self.reslayer = ResLayer(2)

    def forward(self, inputs):

        output_1 = nn.functional.relu(self.layer_1(inputs))

        output_2 = self.layer_2(output_1)
        output_3 = self.layer_3(output_1)

        output_4, output_5 = self.listlayer(output_2, output_3)
        
        output = self.reslayer(output_2, output_4, True)

        return output["logits"]


class CustomBertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_embeddings = BertEmbeddings(config)
        self.bert_layer = BertLayer(config)
        self.bert_output = BertOutput(config)
        self.pooler = BertPooler(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, inputs):
        out = self.bert_embeddings(inputs)
        out = self.bert_layer(out)[0]
        out = self.bert_output(out, out)
        pool_output = self.pooler(out)

        return self.classifier(pool_output)


if __name__ == "__main__":
    # dim = 10
    # data = torch.randn(30, dim)
    # dset = TensorDataset(data)

    # num_samples = 100

    # m = ResModel(dim)

    # fisher = compute_fisher(m, dset, num_samples)

    # # print(fisher)

    # fisher_inferonce = compute_fisher_inferonce(m, dset, num_samples, 30)

    # # print(fisher_inferonce)

    # batch_fisher = compute_fisher_batchwise(m, dset, num_samples, 30, 2)

    # # print(batch_fisher)


    # print(fisher)
    # print("===")
    # print(batch_fisher)

    # for name, value in fisher.items():
    #     match = torch.allclose(value, batch_fisher[name], atol=1e-7)
    #     print("{:<20} {}".format(name, match))

    config = {
        "attention_probs_dropout_prob": 0.0,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 16,
        "initializer_range": 0.02,
        "intermediate_size": 16,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.3.2",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 10,
        "num_labels": 2
    }

    config = BertConfig.from_dict(config)

    m = AutoModelForSequenceClassification.from_config(config)

    dim = 10
    data = torch.randint(0, 10, (dim, 5))

    dset = TensorDataset(data)

    num_samples = 100

    m.to(cuda_device)

    fisher = compute_fisher(m, dset, num_samples)

    # print(fisher)

    fisher_inferonce = compute_fisher_inferonce(m, dset, num_samples, 30)

    # print(fisher_inferonce)

    batch_fisher = compute_fisher_batchwise(m, dset, num_samples, 10, 2)


    opacus_fisher = compute_fisher_opacus(m, dset, num_samples, 10, 2)

    # print(batch_fisher)

    # print(fisher)
    # print("===")
    # print(batch_fisher)

    for name, value in fisher.items():
        match = torch.allclose(value, opacus_fisher[name], atol=1e-7)
        print("{:<20} {}".format(name, match))

        if not match:
            print(name)
            print(value)
            print(opacus_fisher[name])
