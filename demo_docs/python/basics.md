# Python functions and comprehensions

A function is defined with the `def` keyword followed by its name and a
parameter list. Calling a function evaluates its body and returns the value
from the first executed `return` statement. A function without an executed
`return` statement returns `None`.

List comprehensions provide a concise way to create lists. Their basic form is
`[expression for item in iterable if condition]`. The optional condition filters
items before the expression is added to the resulting list.

Generators use `yield` to produce a sequence lazily. Calling a generator
function returns an iterator; execution resumes after `yield` when the next
value is requested.
