import copy
from typing import List, Any

import torch
from torch import nn
from torch import Tensor
import torchdynamo
import alpa
import alpa.torch as atorch
import jax
from torch.fx.experimental.normalize import NormalizeOperators
from torchdynamo.optimizations.normalize import normalize, DONT_EXPAND_MODULES

from alpa.torch.utils import make_shaped_array_from_pt_tensor

decomp_prefix = "alpa_torch_ops_decomposition"
infer_prefix = "alpa_torch_ops_infer_shape"
mapping_prefix = "alpa_torch_ops_mapping"

# Copied from functorch/functorch/_src/make_functional.py
def _del_nested_attr(obj: nn.Module, names: List[str]) -> None:
    """
    Deletes the attribute specified by the given list of names.
    For example, to delete the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'])
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj: nn.Module, names: List[str], value: Tensor) -> None:
    """
    Set the attribute specified by the given list of names to value.
    For example, to set the attribute obj.conv.weight,
    use _del_nested_attr(obj, ['conv', 'weight'], value)
    """
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def _get_nested_attr(obj: nn.Module, names: List[str]) -> None:
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        _get_nested_attr(getattr(obj, names[0]), names[1:])


def _swap_state(mod: nn.Module, names_map: List[str], elems):
    result = []
    for (_, attr_names), elem in zip(names_map.items(), elems):
        for i, attr_name in enumerate(attr_names):
            if i == 0:
                result.append(_get_nested_attr(mod, attr_name))
            _del_nested_attr(mod, attr_name)
            _set_nested_attr(mod, attr_name, elem)
    return result


def create_names_map(named_params, tied_named_params):
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
    named_params = {k: v for k, v in named_params}
    tied_named_params = {k: v for k, v in tied_named_params}

    tensors_dict_keys = set(named_params.keys())
    tied_tensors_dict_keys = set(tied_named_params.keys())
    assert tensors_dict_keys.issubset(tied_tensors_dict_keys)

    tensor_to_mapping = {}
    for key, tensor in named_params.items():
        tensor_to_mapping[tensor] = (key, [])
    for key, tensor in tied_named_params.items():
        assert tensor in tensor_to_mapping
        tensor_to_mapping[tensor][1].append(key.split('.'))
    result = {key: value for key, value in tensor_to_mapping.values()}
    return result


def _extract_members(mod: nn.Module, _named_members, named_members, subclass):
    all_named_members = tuple(_named_members(mod, remove_duplicate=False))
    named_members = tuple(named_members())
    names_map = create_names_map(named_members, all_named_members)

    # Remove all the members in the model
    memo = {}
    for name, p in all_named_members:
        if p not in memo:
            memo[p] = subclass(torch.empty_like(p, device='meta'))
        replacement = memo[p]
        _set_nested_attr(mod, name.split("."), replacement)

    if len(named_members) == 0:
        names, params = (), ()
    else:
        names, params = zip(*named_members)
    return params, names, names_map


def extract_weights(mod: nn.Module):
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
    return _extract_members(mod, _named_parameters, mod.named_parameters, nn.Parameter)


def extract_buffers(mod: nn.Module):
    return _extract_members(mod, _named_buffers, mod.named_buffers, lambda x: x)


def fx_ir_to_alpa_func_code(fx_ir, alpa_func_name):
    # TODO: maybe we can operate on FX IR node to make this function impl cleaner

    fx_ir_code_cleaned = ""
    for line in fx_ir.code.strip().split("\n"):
        line = line.replace(";  ", "\n    ")
        fx_ir_code_cleaned += (line + "\n")

    if atorch.debug:
        print("fx_ir_code_cleaned: ")
        print(fx_ir_code_cleaned)

    lines = fx_ir_code_cleaned.split("\n")
    assert "def forward(" in lines[0]
    signature_line = lines[0]
    sig_args = signature_line.split("def forward(")[1].split("):")[0].split(", ")
    sig_args = sig_args[1:]  # remove `self`
    first_input_arg = sig_args[0]
    sig_args.insert(0, "params")
    sig_args.insert(1, "bufs")
    signature_line = f"def {alpa_func_name}(" + ", ".join(sig_args) + "):"

    out_body_lines = []

    bufs_set = set(fx_ir.buffers(recurse=True))
    bufs_name_to_key_mapping = {}

    def getattr_recursive(module, attr_fqn):
        attr_path_segments = attr_fqn.split(".")
        ret = module
        for seg in attr_path_segments:
            ret = getattr(ret, seg)
        return ret

    for line in lines[1:]:
        line = line.replace(" : torch.Tensor", "")
        if "self." in line:
            attr_name = line.split("self.")[1].split("(")[0]
            attr_value = getattr_recursive(fx_ir, attr_name)
            if isinstance(attr_value, torch.nn.Module):
                assert attr_value.__class__.__name__ in DONT_EXPAND_MODULES, "unknown module: " + str(attr_value.__class__.__name__)
                call_args = line.split("self.")[1].split("(")[1].split(")")[0].split(", ")
                # Full list of NN modules that need this handling is at torchdynamo/torchdynamo/optimizations/normalize.py `DONT_EXPAND_MODULES`.
                if attr_value.__class__.__name__ == "Conv2d":
                    call_args += [
                        f"params['{attr_name}.weight']",
                        f"bias=params['{attr_name}.bias']",
                        f"stride={attr_value.stride}",
                        f"padding={attr_value.padding}",
                        f"dilation={attr_value.dilation}",
                        f"groups={attr_value.groups}",
                    ]
                    lhs = line.split(" = ")[0]
                    line = lhs + " = " + f"torch.conv2d({', '.join(call_args)})"
                else:
                    raise NotImplementedError
            elif isinstance(attr_value, torch.nn.Parameter):  # Parameter
                line = line.replace(f"self.{attr_name}", f"params['{attr_name}']")
            elif isinstance(attr_value, torch.Tensor):
                if attr_value in bufs_set:  # Buffer
                    # NOTE: torchdynamo somehow puts both buffer and non-buffer Tensors
                    # (i.e. both `self.register_buffer(...)` and `self.tensor = torch.tensor(...)`)
                    # into buffers dict, which is a deviation from PT eager mode contract :(
                    line = line.replace(f"self.{attr_name}", f"bufs['{attr_name}']")
                else:  # Const
                    raise ValueError(
                        f"We assume torchdynamo treats non-buffer tensor attributes as buffers, " + \
                        "but this assumption no longer holds true for .{attr_name}"
                    )
                    # line = line.replace(f"self.{attr_name}", f"consts['{attr_name}']")
            else:  # Const
                raise ValueError(f"non-module / non-tensor attribute is not supported, but found type of .{attr_name} to be {type(attr_value)}")
                # line = line.replace(f"self.{attr_name}", f"consts['{attr_name}']")

        # Record all buffers' name and their correponding key in `bufs` dict
        if " = bufs['" in line:
            buf_name = line.split(" = bufs['")[0].strip()
            buf_key = line.split(" = bufs['")[1].split("']")[0]
            bufs_name_to_key_mapping[buf_name] = buf_key

        # Rewrite stateful modules / ops
        if "torch.nn.functional.batch_norm" in line:
            lhs = line.split(" = torch.nn.functional.batch_norm")[0]
            call_args = line.split(" = torch.nn.functional.batch_norm(")[1].split(")")[0].split(", ")
            running_mean_arg_name = call_args[1]
            assert "running_mean" in running_mean_arg_name
            running_var_arg_name = call_args[2]
            assert "running_var" in running_var_arg_name
            line = lhs + ", running_mean_new, running_var_new" + " = torch.nn.functional.batch_norm(" + ", ".join(call_args) + ")"
            line += "\n"
            line += f"    bufs['{bufs_name_to_key_mapping[running_mean_arg_name]}'] = running_mean_new"
            line += "\n"
            line += f"    bufs['{bufs_name_to_key_mapping[running_var_arg_name]}'] = running_var_new"

        # Op lowering
        if "torch._C._nn." in line:
            op_name = line.split("torch._C._nn.")[1].split("(")[0]
            line = line.replace(f"torch._C._nn.{op_name}", f"torch.nn.functional.{op_name}")
        if f"{infer_prefix}_" in line:
            op_name = line.split(f"{infer_prefix}_torch_")[1].split("(")[0]
            line = line.replace(f"{infer_prefix}_torch_{op_name}", f"torch.{op_name}")
        if f"{decomp_prefix}_torch_nn_functional_" in line:
            op_name = line.split(f"{decomp_prefix}_torch_nn_functional_")[1].split("(")[0]
            line = line.replace(f"{decomp_prefix}_torch_nn_functional_{op_name}", f"torch.nn.functional.{op_name}")
        if f"{decomp_prefix}_torch_" in line:
            op_name = line.split(f"{decomp_prefix}_torch_")[1].split("(")[0]
            line = line.replace(f"{decomp_prefix}_torch_{op_name}", f"torch.{op_name}")
        if f"{mapping_prefix}_torch_nn_functional_" in line:
            op_name = line.split(f"{mapping_prefix}_torch_nn_functional_")[1].split("(")[0]
            line = line.replace(f"{mapping_prefix}_torch_nn_functional_{op_name}", f"torch.nn.functional.{op_name}")
        if f"{mapping_prefix}_torch_" in line:
            op_name = line.split(f"{mapping_prefix}_torch_")[1].split("(")[0]
            line = line.replace(f"{mapping_prefix}_torch_{op_name}", f"torch.{op_name}")
        if ".dim()" in line:
            tensor_name = line.split(" = ")[1].split(".dim()")[0]
            line = line.replace(f"{tensor_name}.dim()", f"len({tensor_name}.shape)")
        if ".permute(" in line:
            tensor_name = line.split(" = ")[1].split(".permute(")[0]
            line = line.replace(f"{tensor_name}.permute(", f"torch.permute({tensor_name}, (") + ")"
        if ".expand(" in line:
            tensor_name = line.split(" = ")[1].split(".expand(")[0]
            line = line.replace(f"{tensor_name}.expand(", f"torch.expand({tensor_name}, (") + ")"
        if ".view(" in line:
            tensor_name = line.split(" = ")[1].split(".view(")[0]
            line = line.replace(f"{tensor_name}.view(", f"torch.view({tensor_name}, (") + ")"
        if " @ " in line:
            lhs = line.split(" = ")[0]
            rhs = line.split(" = ")[1]
            line = lhs + " = " + "torch.matmul(" + rhs.replace(" @ ", ", ") + ")"

        if "return " in line:
            rhs_of_return = line.split("return (")[1]
            then_lhs_of_right_parenthesis = rhs_of_return.split(")")[0]
            assert "(" not in then_lhs_of_right_parenthesis, "Nested tuple is not supported in output yet."
            output_args = then_lhs_of_right_parenthesis.split(",")
            output_args.insert(0, "bufs")
            line = line.split("return (")[0] + "return " + ", ".join(output_args)

        out_body_lines.append(line)

    jax_func_code = signature_line + "\n" + "\n".join(out_body_lines) + "\n"
    jax_func_code = jax_func_code.strip()

    return jax_func_code


# Copied from functorch/functorch/_src/named_members_polyfill.py
def _named_members(mod, get_members_fn, prefix='', recurse=True):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    modules = mod.named_modules(prefix=prefix, remove_duplicate=False) if recurse else [(prefix, mod)]
    for module_prefix, module in modules:
        members = get_members_fn(module)
        for k, v in members:
            if v is None or v in memo:
                continue
            if v in memo:
                raise ValueError(f"Weight-sharing is not supported by Alpa, but we found duplicated parameter / buffer in model: {k}")
            else:
                memo.add(v)
            name = module_prefix + ('.' if module_prefix else '') + k
            yield name, v


def _named_parameters(mod, prefix: str = '', recurse: bool = True):
    gen = _named_members(
        mod,
        lambda module: module._parameters.items(),
        prefix=prefix, recurse=recurse)
    for elem in gen:
        yield elem


def _named_buffers(mod, prefix: str = '', recurse: bool = True):
    gen = _named_members(
        mod,
        lambda module: module._buffers.items(),
        prefix=prefix, recurse=recurse)
    for elem in gen:
        yield elem


# Copied from torchdynamo/torchdynamo/optimizations/normalize.py
def normalize_ir_no_run(fx_ir):
    normalize(fx_ir)
    try:
        fx_ir = NormalizeOperators(fx_ir).transform()
    except AttributeError:
        # log.exception("NormalizeOperators() failed")
        pass
    # ShapeAliasingAndMutationProp(fx_ir).run(*example_inputs)
    # fx_ir = Functionalization(fx_ir).transform()
    fx_ir.recompile()
    # record_graph_stats(fx_ir)
    return fx_ir


# Adapted from `FunctionalModuleWithBuffers` in functorch/functorch/_src/make_functional.py
class FunctionalModuleWithBuffersInInputAndOutput(torch.nn.Module):
    """
    Given a ``torch.nn.Module``, `create_from` extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    Compared to `FunctionalModuleWithBuffers` in functorch, the returned functional version
    of the model also has buffers in the output, since buffer values can be changed with
    operations like batchnorm and should be tracked as part of output.
    """
    def __init__(self, stateless_model, param_names, buffer_names,
                 param_names_map, buffer_names_map):
        super(FunctionalModuleWithBuffersInInputAndOutput, self).__init__()
        self.stateless_model = stateless_model
        self.param_names = param_names
        self.buffer_names = buffer_names

        self.all_names_map = dict(param_names_map)
        self.all_names_map.update(buffer_names_map)

    @staticmethod
    def create_from(model, disable_autograd_tracking=False):
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = copy.deepcopy(model)
        param_values, param_names, param_names_map = extract_weights(model_copy)
        buffer_values, buffer_names, buffer_names_map = extract_buffers(model_copy)
        params = dict(zip(param_names, param_values))
        buffers = dict(zip(buffer_names, buffer_values))
        if disable_autograd_tracking:
            for param in params_values:
                param.requires_grad_(False)
        return (
            FunctionalModuleWithBuffersInInputAndOutput(model_copy, param_names, buffer_names,
                                        param_names_map, buffer_names_map),
            params,
            buffers,
        )

    def forward(self, params, buffers, *args, **kwargs):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(
            self.stateless_model,
            self.all_names_map,
            list(params.values()) + list(buffers.values()))
        try:
            return buffers, self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            _swap_state(self.stateless_model, self.all_names_map, old_state)


def functionalize(module: torch.nn.Module, inputs, batch_dim, num_micro_batches):
    """
    Returns:
        - `module_func`: a function that has same logic as x.forward but callable with either PT or Alpa inputs. It:
            - wraps the original inputs in a tuple
            - takes `params` and `bufs` as extra input at beginning of input list
            - produces `bufs` as extra output at beginning of output list
            - all calls are made compatible with Alpa, e.g.:
                - replaces all unexpandable module calls (e.g. nn.Conv2d) with equivalent `torch.*` function calls
                - replaces all torch.nn.functional calls that has in-place ops (e.g. F.batch_norm) with equivalent `alpa.torch.*` function calls that has buffer as part of output
                - complex torch function calls (e.g. F.dropout) are decomposed and implemented with `torch.*` calls
        - `params`: a dict of shape-only tensors representing the trainable parameters of the module. In PT format if "local", in Alpa format if "dist".
        - `bufs`: a dict of shape-only tensors representing the no-gradient parameters of the module. In PT format if "local", in Alpa format if "dist".
    Throws error if x.forward:
        - has in-place ops
        - or, has data-dependent control flow
        - or, has other graph-breaking statements (e.g. `print()`) that prevents the program from being captured as a single graph (only in "dist" mode)
    """

    # This param/buffer name map is used for mapping from FQN in original PyTorch model to FQN in PyTorch FX IR.
    tensor_to_name_map = {}

    all_tensors_pt_orig = dict(_named_parameters(module))
    all_tensors_pt_orig.update(dict(_named_buffers(module)))

    for k, v in all_tensors_pt_orig.items():
        assert v not in tensor_to_name_map
        tensor_to_name_map[v] = {"orig_name": k}

    def add_transformed_name(tensor_to_name_map, k, v):
        assert v in tensor_to_name_map
        assert "transformed_name" not in tensor_to_name_map[v]
        tensor_to_name_map[v]["transformed_name"] = k

    if alpa.torch.mode() == "dist":
        # In dist mode, use TorchDynamo to enforce:
        # 1) no data-dependent control flow
        # 2) no graph break points
        # 3) no in-place ops

        # `torch.fx.symbolic_trace` gives good error message for data-dependent control flow.
        # It's debatable whether we should pass output of FX-trace (i.e. a FX-traced module) into torchdynamo, so not doing that right now.
        _ = torch.fx.symbolic_trace(module)

        def convert_pt_module_to_jax_func(module, inputs):
            fx_ir = None

            def process_graph(fx_ir_captured: torch.fx.GraphModule, example_inputs_pt: List[torch.Tensor]):
                # NOTE: `fx_ir_captured` is only the forward pass of PyTorch model.
                nonlocal fx_ir
                if atorch.debug:
                    print(f"fx_ir_captured.code: {fx_ir_captured.code}")
                fx_ir = fx_ir_captured
                return fx_ir_captured.forward  # return a python callable

            torchdynamo.config.debug = False
            # During trace time, we apply both `decompose_ops` and `infer_output_shape` context managers,
            # to both decompose the large PyTorch ops into smaller ones and add meta tensor support to ops as needed.
            with alpa.torch.ops.decompose_ops():
                with alpa.torch.ops.infer_output_shape():
                    # `torchdynamo` is good at tracing ops at `torch.*` level. TODO add more reasoning.
                    with torchdynamo.optimize(process_graph, nopython=True):
                        # This call populates the global `fx_ir` variable
                        module(*inputs)

            assert fx_ir is not None
            fx_ir = normalize_ir_no_run(fx_ir)

            # NOTE: due to some unknown reason, only the second normalize pass
            # can convert tensor method to torch function (e.g. `.t()` to `torch.t()`)
            fx_ir = normalize_ir_no_run(fx_ir)

            module_func_name = "forward_func"
            module_func_code = fx_ir_to_alpa_func_code(fx_ir, module_func_name)

            if atorch.debug:
                print("module_func_code: ")
                print(module_func_code)

            exec(module_func_code)
            module_func = locals()[module_func_name]

            return fx_ir, module_func

        # NOTE: Unfortunately PyTorch doesn't have "symbolic batch size" for now, and both batch size and "-1" size are hardcoded in FX IR.
        # As a result, we need to generate two graphs (one with full-batch size, one with micro-batch size) to be used by Alpa.
        fx_ir_full_batch, module_func_full_batch = convert_pt_module_to_jax_func(module, inputs)
        if num_micro_batches > 1:
            micro_batch_size = inputs[0].shape[batch_dim] // num_micro_batches
            _, module_func_micro_batch = convert_pt_module_to_jax_func(module, [inputs[0].narrow(batch_dim, 0, micro_batch_size)])

        def module_func(*inputs):
            if num_micro_batches > 1:
                # functionalized module input signature is (params, bufs, x), so actual input is at index 2.
                if inputs[2].shape[batch_dim] == micro_batch_size:
                    return module_func_micro_batch(*inputs)
                elif inputs[2].shape[batch_dim] > micro_batch_size:
                    return module_func_full_batch(*inputs)
            else:
                return module_func_full_batch(*inputs)

        params_pt = dict(_named_parameters(fx_ir_full_batch))
        bufs_pt = dict(_named_buffers(fx_ir_full_batch))

        for k, v in params_pt.items():
            add_transformed_name(tensor_to_name_map, k, v)

        for k, v in bufs_pt.items():
            add_transformed_name(tensor_to_name_map, k, v)

        for k in tensor_to_name_map:
            if "transformed_name" not in tensor_to_name_map[k]:
                print(tensor_to_name_map[k]["orig_name"])

        params_alpa = {k: make_shaped_array_from_pt_tensor(v) for k,v in params_pt.items()}
        bufs_alpa = {k: make_shaped_array_from_pt_tensor(v) for k,v in bufs_pt.items()}

        if alpa.torch.mode() == "local":
            params = params_pt
            bufs = bufs_pt
        elif alpa.torch.mode() == "dist":
            params = params_alpa
            bufs = bufs_alpa

        name_map = {}
        for elem in tensor_to_name_map.values():
            try:
                name_map[elem["orig_name"]] = elem["transformed_name"]
            except KeyError as e:
                print(f'elem["orig_name"]: {elem["orig_name"]}')
                raise e
        return module_func, params, bufs, name_map
    elif alpa.torch.mode() == "local":
        # In local mode, use functionalization pass adapted from functorch
        # TODO: add more rigorous unit tests for this branch
        module_func, params, bufs = FunctionalModuleWithBuffersInInputAndOutput.create_from(module)
        name_map = {}
        for elem in tensor_to_name_map.values():
            name_map[elem["orig_name"]] = elem["orig_name"]
        return module_func, params, bufs, name_map
