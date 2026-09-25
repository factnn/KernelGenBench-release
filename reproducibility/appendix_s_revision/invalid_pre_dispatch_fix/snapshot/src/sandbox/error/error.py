def register_error(e):
    raise RuntimeError(
        e, "An error was encountered while registering the triton operator."
    )
