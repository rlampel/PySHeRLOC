import importlib


def get_problem(problem_name):
    """Return instance of the problem class with the matching name.

    Keyword arguments:
        problem_name    -- name of the problem
    """
    curr_name = "Apps."
    for character in problem_name:
        if character == " ":
            curr_name += "_"
        elif character == "'":
            pass
        else:
            curr_name += character

    try:
        curr_module = importlib.import_module(curr_name)
    except ModuleNotFoundError:
        raise ValueError("This problem does not exist.")

    return curr_module.problem()


def get_oed_problem(problem_name, criterion="A"):
    """Return instance of the problem class with the matching name.

    Keyword arguments:
        problem_name    -- name of the problem
    """
    curr_name = "Apps.oed."
    for character in problem_name:
        if character == " ":
            curr_name += "_"
        elif character == "'":
            pass
        else:
            curr_name += character

    try:
        curr_module = importlib.import_module(curr_name)
    except ModuleNotFoundError:
        raise ValueError("This problem does not exist.")

    return curr_module.problem(criterion)
