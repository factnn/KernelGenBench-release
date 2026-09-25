# import itertools

# _parameter_registry = {}

# def parametrize(param_names, values):
#     def decorator(func):
#         name = func.__name__
#         if name not in _parameter_registry:
#             _parameter_registry[name] = {}

#         # Parse param_names, supports "a" or "a,b,c"
#         if isinstance(param_names, str):
#             param_names_list = [p.strip() for p in param_names.split(",")]
#         else:
#             raise ValueError("param_names must be a string")

#         # If there is only one parameter name, store the corresponding values directly
#         if len(param_names_list) == 1:
#             _parameter_registry[name][param_names_list[0]] = values
#         else:
#             # Multiple parameter names: values should be an iterable list of tuples
#             # Split values by position and store as lists for each parameter name
#             transposed = list(zip(*values))  # convert [(a1,b1,c1), (a2,b2,c2)] to [(a1,a2), (b1,b2), (c1,c2)]
#             for i, param in enumerate(param_names_list):
#                 _parameter_registry[name][param] = list(transposed[i])

#         return func
#     return decorator

# def get_registry():
#     return _parameter_registry

# label_registry.py
_label_registry = {}

def label(name):
    """Add a label to a test function"""
    def decorator(func):
        # breakpoint()
        _label_registry.setdefault(name, []).append((func, "default"))
        return func
    return decorator

def get_funcs_by_label(name):
    """Get all functions by label"""
    return _label_registry.get(name, [])

def get_all_labels():
    return list(_label_registry.keys())


_parameter_registry = {}

class Param:
    def __init__(self, *values, marks: str | list[str] = None):
        self.values = [values]
        self.marks = marks if isinstance(marks, list) else [marks]

    def __repr__(self) -> str:
        return f"Param(values={self.values}, marks={self.marks})"

def parametrize(param_names, values):
    def decorator(func):
        name = func.__name__
        if name not in _parameter_registry:
            _parameter_registry[name] = {}

        if isinstance(param_names, str):
            param_names_list = [p.strip() for p in param_names.split(",")]
        else:
            raise ValueError("param_names must be a string")

        if isinstance(values[0], Param):
            for param in values:
                for mark in param.marks:
                    _parameter_registry[name].setdefault(mark, []).append(
                        {
                            "names": param_names_list,
                            "values": param.values
                        }
                    )
                    _label_registry.setdefault(mark, []).append((func, mark))
        else:
            # slot: a set of parameter names and corresponding values
            # _parameter_registry[name].append({
            #     "names": param_names_list,
            #     "values": values
            # })
            _parameter_registry[name].setdefault("default", []).append({
                "names": param_names_list,
                "values": values
            })

        return func
    return decorator

def get_registry():
    return _parameter_registry

def get_params(name, mark="default"):
    return _parameter_registry.get(name, {}).get(mark, {})
