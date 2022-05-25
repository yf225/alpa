import copy
from typing import List, Any

import torch
import torchdynamo
import functorch
import alpa
import jax
from torch.fx.experimental.normalize import NormalizeOperators
from torchdynamo.optimizations.normalize import normalize, DONT_EXPAND_MODULES
from functorch._src.make_functional import extract_weights, extract_buffers, _swap_state

from alpa.torch.utils import make_shaped_array_from_pt_tensor

decomp_prefix = "alpa_torch_ops_decomposition"
infer_prefix = "alpa_torch_ops_infer_shape"


def fx_ir_to_alpa_func_code(fx_ir, alpa_func_name):
    # TODO: maybe we can operate on FX IR node to make this function impl cleaner

    fx_ir_code_cleaned = ""
    for line in fx_ir.code.strip().split("\n"):
        line = line.replace(";  ", "\n    ")
        fx_ir_code_cleaned += (line + "\n")

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
    out_body_lines.append("    alpa = globals()['alpa']")

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
            line = line.replace(f"torch._C._nn.{op_name}", f"torch.{op_name}")
        if f"{infer_prefix}_" in line:
            op_name = line.split(f"{infer_prefix}_torch_")[1].split("(")[0]
            line = line.replace(f"{infer_prefix}_torch_{op_name}", f"torch.{op_name}")
        if f"{decomp_prefix}_torch_nn_functional_" in line:
            op_name = line.split(f"{decomp_prefix}_torch_nn_functional_")[1].split("(")[0]
            line = line.replace(f"{decomp_prefix}_torch_nn_functional_{op_name}", f"torch.nn.functional.{op_name}")
        if f"{decomp_prefix}_torch_" in line:
            op_name = line.split(f"{decomp_prefix}_torch_")[1].split("(")[0]
            line = line.replace(f"{decomp_prefix}_torch_{op_name}", f"torch.{op_name}")
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

"""
alpa.torch.nn.make_func(x: torch.nn.Module)
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
def make_func(module: torch.nn.Module, *inputs):
    module_func = None
    params = None
    bufs = None

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

    def process_graph(fx_ir: torch.fx.GraphModule, example_inputs_pt: List[torch.Tensor]):
        # NOTE: `fx_ir` is only the forward pass of PyTorch model.

        nonlocal module_func
        nonlocal params
        nonlocal bufs
        nonlocal tensor_to_name_map

        fx_ir_normalized = normalize_ir_no_run(fx_ir)

        # NOTE: due to some unknown reason, only the second normalize pass
        # can convert tensor method to torch function (e.g. `.t()` to `torch.t()`)
        fx_ir_normalized = normalize_ir_no_run(fx_ir_normalized)

        module_func_name = "forward_func"
        module_func_code = fx_ir_to_alpa_func_code(fx_ir_normalized, module_func_name)

        print("module_func_code: ")
        print(module_func_code)

        exec(module_func_code)
        module_func = locals()[module_func_name]

        params_pt = dict(_named_parameters(fx_ir_normalized))
        bufs_pt = dict(_named_buffers(fx_ir_normalized))

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

        return fx_ir_normalized.forward  # return a python callable

    if alpa.torch.mode() == "dist":
        # In dist mode, use TorchDynamo to enforce:
        # 1) no data-dependent control flow
        # 2) no graph break points
        # 3) no in-place ops

        # `torch.fx.symbolic_trace` gives good error message for data-dependent control flow.
        # It's debatable whether we should use output of FX-trace as input into torchdynamo, so not doing that right now.
        _ = torch.fx.symbolic_trace(module)

        torchdynamo.config.debug = False
        # During trace time, we apply both `decompose_ops` and `infer_output_shape` context managers,
        # to both decompose the large PyTorch ops into smaller ones and add meta tensor support to ops as needed.
        with alpa.torch.ops.decompose_ops():
            with alpa.torch.ops.infer_output_shape():
                # `torchdynamo` is good at tracing ops at `torch.*` level. TODO add more reasoning.
                with torchdynamo.optimize(process_graph, nopython=True):
                    module(*inputs)
        name_map = {}
        for elem in tensor_to_name_map.values():
            try:
                name_map[elem["orig_name"]] = elem["transformed_name"]
            except KeyError as e:
                print(f'elem["orig_name"]: {elem["orig_name"]}')
                raise e
    elif alpa.torch.mode() == "local":
        # In local mode, use functionalization pass adapted from functorch
        # TODO: add more rigorous unit tests for this branch
        module_func, params, bufs = FunctionalModuleWithBuffersInInputAndOutput.create_from(module)
        name_map = {}
        for elem in tensor_to_name_map.values():
            name_map[elem["orig_name"]] = elem["orig_name"]

    return module_func, params, bufs, name_map
