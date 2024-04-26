from sympy import *
import warnings
def expr_to_parsable(expr, dict):
    mirrorred_expr = expr.subs(dict)
    return mirrorred_expr.__str__()

def substitute_function(expr, g):
    '''Give a function (or variable) that needs to be replaced
       in any instance of a  derivative in an expression.
       A version of this code with different lists of functions
       (or a filter on the order of the derivative) should be written.
    '''
    if list(expr.atoms(Function)) == []:
        warnings.warn("Er is geen functie om te vervangen...")
        return expr, None
    # simply find the first instance of a function, and save what it is:
    func = expr.atoms(Function).pop()
    # argske = func.args[0]
    return expr.subs(func, g).doit(), func

def substitute_derivative(expr, df):
    '''Give a function (or variable) that needs to be replaced
       in any instance of a  derivative in an expression.
       A version of this code with different lists of functions
       (or a filter on the order of the derivative) should be written.
    '''
    if list(expr.atoms(Derivative)) == []:
        warnings.warn("Er is geen afgeleide om te vervangen...")
        return expr, None, None

    # simply find the first instance of a derivative:
    deriv = expr.atoms(Derivative).pop()
    func = deriv.args[0]
    dummy_var = deriv.args[1][0]

    # df obviously cannot depend on the dummy variable, whether it be a function or a variable.
    return expr.subs(func, dummy_var * df).doit(), func, dummy_var