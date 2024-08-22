def optimize(x0,func,eps=0.1):
    epsilon=1
    def derivative(f,x,e=0.01):
        return (f(x+e)-f(x-e))/(2*e)
    def second_der(f,x,e=0.01):
        return (f(x+e)-2*f(x)+f(x-e))/e**2
    while epsilon>eps:
        x=x0-derivative(func,x0,e=0.01)/second_der(func,x0,e=0.01)
        epsilon=x-x0
        x0=x
    return x0