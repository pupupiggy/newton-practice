def optimize(x0, func, eps=0.1):
    """
    Newton’s method for optimization

    x0: start value
    func: function
    eps: epsilon to stop the loop
    """

    epsilon = 1

    def derivative(f, x, e=0.01):
        """
        Estimate the first derivative of f at x using the finite difference method.

        f: objective function
        x: variable value
        e: step
        """
        return (f(x + e) - f(x - e)) / (2 * e)

    def second_der(f, x, e=0.01):
        """
        Estimate the second derivative of f at x using the finite difference method.

        f: objective function
        x: variable value
        e: step
        """
        return (derivative(f, x + e, e) - derivative(f, x, e)) / e

    while epsilon > eps:  # Stop loop when epsilon is very small
        x = x0 - derivative(func, x0, e=0.01) / second_der(
            func, x0, e=0.01
        )  # Calculate new x
        epsilon = abs(x - x0)  # Calculate delta (use abs to ensure positive value)
        x0 = x  # Iterate x

    return x0




##hello world
