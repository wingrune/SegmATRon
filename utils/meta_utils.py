# This code is copied from https://github.com/allenai/interactron
import collections
import torch


def get_parameters(model):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        #print("Model is last child")
        params = []
        for name, param in model._parameters.items():
            #print(name)
            #print(param)
            #exit()
            #print(param.requires_grad)
            #exit()
            if param is not None and param.requires_grad:
                params.append(param)
                #print("Appended param")
        return params
        # return [child for child in model._parameters.values() if child.requires_grad]
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_parameters(child))
            except TypeError:
                flatt_children.append(get_parameters(child))
    #print(flatt_children)
    #exit()
    return tuple(flatt_children)


# def detach_parameters(model):
#     # get children form model!
#     children = list(model.children())
#     flatt_children = []
#     if children == []:
#         # if model has no children; model is last child! :O
#         for name, param in model._parameters.items():
#             if param is not None and param.requires_grad:
#                 model._parameters[name] = model._parameters[name].detach()
#                 model._parameters[name].requires_grad = True
#                 # return [child for child in model._parameters.values() if child.requires_grad]
#     else:
#         # look for children from children... to the last child!
#         for child in children:
#             try:
#                 flatt_children.extend(detach_parameters(child))
#             except TypeError:
#                 flatt_children.append(detach_parameters(child))
#     return flatt_children


def detach_parameters(params):
    detached_params = []
    for p in params:
        dp = p.clone().detach()
        dp.requires_grad = True
        detached_params.append(dp)
    return tuple(detached_params)


def detach_gradients(params):
    detached_params = []
    for p in params:
        if p is None:
            detached_params.append(None)
        else:
            dp = p.clone().detach()
            detached_params.append(dp)
    return tuple(detached_params)


def set_parameters(model, params):
    # get children form model!
    if not isinstance(params, collections.abc.Iterator):
        params = iter(params)
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        for name, param in model._parameters.items():
            if param is not None and param.requires_grad:
                model._parameters[name] = next(params)
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(set_parameters(child, params))
            except TypeError:
                flatt_children.append(set_parameters(child, params))
    return flatt_children


# def clone_parameters(model, params):
#     # get children form model!
#     children = list(model.children())
#     flatt_children = []
#     if children == []:
#         # if model has no children; model is last child! :O
#         for name, param in model._parameters.items():
#             if param is not None and param.requires_grad:
#                 model._parameters[name] = next(params).clone()
#     else:
#         # look for children from children... to the last child!
#         for child in children:
#             try:
#                 flatt_children.extend(clone_parameters(child, params))
#             except TypeError:
#                 flatt_children.append(clone_parameters(child, params))
#     return flatt_children

def clone_parameters(params):
    cloned_parameters = []
    for param in params:
        cloned_parameters.append(param.clone())
    return tuple(cloned_parameters)


# def sgd_step(model, grads, lr=0.001):
#     # get children form model!
#     children = list(model.children())
#     flatt_children = []
#     if children == []:
#         # if model has no children; model is last child! :O
#         for name, param in model._parameters.items():
#             if param is not None and param.requires_grad:
#                 grad = next(grads)
#                 if grad is not None:
#                     model._parameters[name] = model._parameters[name] - lr * grad
#     else:
#         # look for children from children... to the last child!
#         for child in children:
#             try:
#                 flatt_children.extend(sgd_step(child, grads, lr=lr))
#             except TypeError:
#                 flatt_children.append(sgd_step(child, grads, lr=lr))
#     return flatt_children


def sgd_step(params, grads, lr, clip=0.01):
    updated_params = []
    #max_parameter = 0
    for p, g in zip(params, grads):
        if g is None:
            updated_params.append(p)
        else:
            updated_params.append(p - torch.clip(g * lr, min=-clip, max=clip))
            #max_p = torch.max(torch.abs(torch.clip(g * lr, min=-clip, max=clip)))
            #if max_p > max_parameter:
            #    max_parameter = max_p
    
    #print("Max gradient:", max_parameter)
    return tuple(updated_params)
