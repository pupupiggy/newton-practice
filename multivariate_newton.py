from autograd import grad,hessian
import numpy as np
def mul_optimize(x0,func,tol=1e-15,max_iter=1000):
    gradient_func = grad(func)
    hessian_func = hessian(func)
    x = x0
    for i in range(max_iter):
        gradient_val = gradient_func(x)
        hessian_val = hessian_func(x)
        
        # 求解线性系统 H(x) * delta = -grad(f(x))
        delta = np.linalg.solve(hessian_val, -gradient_val)
        
        # 更新变量
        x = x + delta
        
        # 检查收敛性
        if np.linalg.norm(delta) < tol:
            print(f"Converged in {i+1} iterations.")
            return x
        
    print("Did not converge within the maximum number of iterations.")
    return x