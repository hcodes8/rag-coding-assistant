# Context managers

The `with` statement runs a context manager's setup and cleanup protocol.
Objects implement the protocol with `__enter__` and `__exit__`. Cleanup runs
when the suite exits, including when an exception is raised. The built-in
`open()` function returns a context manager, so `with open(path) as file:`
closes the file automatically.

`contextlib.contextmanager` can turn a generator function with one `yield`
into a context manager. Code before `yield` performs setup and code after it
performs cleanup.
